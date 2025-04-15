# import libraries & dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# import data processing and ml component functions
from dashboard.data_loader import load_and_clean_data, prepare_monthly_sales
from dashboard.components.similarity import compute_item_similarity, generate_similarity_heatmap, get_top_similar_items
from dashboard.components.sales_trends import plot_seasonal_sales_trends
from dashboard.components.forecast import forecast_item_backtest, forecast_item_future
from dashboard.components.bundles import generate_bundle_recommendations
from dashboard.components.segmentation import segment_sales_by_review_and_payment
from dashboard.components.customer_insights import get_customer_purchase_history, plot_customer_review_trend
from dashboard.components.timing import calculate_item_timing

# load data & process the data
df = load_and_clean_data()
monthly_sales = prepare_monthly_sales(df)
user_item_matrix = df.pivot_table(index='customerID', columns='item', values='review', aggfunc='mean')
similarity_df = compute_item_similarity(user_item_matrix)
bundle_df = generate_bundle_recommendations(df, similarity_df)
segmented_df = segment_sales_by_review_and_payment(df)
timing_df = calculate_item_timing(monthly_sales)

# dash app with bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True
)
server = app.server

# layout per tab
home_layout = dbc.Container([
    html.H3("Machine Learning vs Fashion Trends"),
    
    html.P("""
        Retail giants are no strangers to trying to predict fashion trends, be ahead of the pack, and capitalize big on emerging markets. Otherwise, there can be major consequences when they fail to keep up with the fashion pace. In 2018, more than a dozen retailers like Sears, Toys-R-Us, Rockport, Brookstone, Claire’s, Nine West, and HH Gregg filed for bankruptcy because they struggled to adapt to frequent shifts in customer expectations and rapidly evolving fashion cycles. 
        To keep up with trends and going against online shopping experience, many retailers would focus their insight on the entire fashion styles, big picture bias, generalized market sentiment, and social media buzz. Although this approach can offer high-level insights, it often overlooks the nuanced performance of individual products.
        This research project aims to address how analysts at big clothes retailers can leverage machine learning methods to provide meaningful insights about clothing trends that pertain specifically to the type of items sold and how they relate to other items. We hope to simplify and streamline the process, demonstrating not only the sales pattern of specific items but also highlighting how items similar to them perform. This way, stores can optimize their product marketing, inventory, and promotion strategies. 
    """), 
    html.H4("LITERATURE REVIEW"),

    html.P("""
        Before we discuss the methods and applications we decided to use, we should first examine previous attempts at solving this problem of trying to predict fashion trends. As the industry heavily relies on being ahead of the curve, there is no shortage of outside applications. One such example comes from the Knowledge Enhanced Recurrent Network (KERN) Model by students at the University of Singapore and the Hong Kong Polytechnic University. Our research seeks to find underlying mutual trends within clothing categories or subcategories over time, but KERN was built to get a better idea of fashion trends overall. The researchers used a large data set from Instagram which contained fine-grained fashion elements (like specific clothing categories, attributes like necklines, and styles) and user information over time. They also displayed said dataset over Google trends to vouch for authenticity.
        The researchers argue traditional methods of fashion prediction often rely on subjective inference and that existing data-driven approaches have studied limited fashion elements with simple patterns. KERN uses a type of encoder-decoder neural network called Long-Short Term Memory, which can understand patterns in sequences of data over time. In addition, the KERN model uses both internal knowledge (such as similarity of trend patterns) and external knowledge (affiliation relations within a fashion taxonomy) to enhance its predictions. It constructs a time series of different fashion items and their popularity over time for specific groups of people, and uses its neural network to read past trend data, summarize it, and predict for the future. They found that the model learned some interesting insights, such as sweaters and turtlenecks trending together, while dresses and sweaters would trend opposite. In addition, it can connect material or detail to specific fashion items to predict their popularity as well. Overall, KERN is an incredibly insightful solution to predicting fashion trends, using time series just like us, but transforming it with a neural network instead of using collaborative filtering.

           """),

    html.H4("ML METHODOLOGY AND DATA COLLECTION"),

    html.P("""
    In order to build our project, we needed a dataset of consumer purchasing patterns, ideally one with sales volumes, clothing types and consumer ratings. Very few datasets we could obtain for free contained all three of these variables, and the ones that did were often paywalled, such as the US ZARA sales dataset. Thus, we settled on a small database we found on Kaggle for the sales reports of a retail shop called Family Fashion Clothing, and used that as our baseline for our project. First, we ensured the dates were changed to be properly usable for time based operations, removed rows where sale amount was missing, and created a “month” column for our later time series analysis. Finally, we removed all items from the dataset that were not explicitly clothing items but had been sold in store, such as bags, wallets, etc. The two machine learning methods that we decided to use were Collaborative Filtering and Time Series analysis. First, create an Item-Item comparison matrix using cosine similarity. 
	In our implementation we aren’t simply comparing ratings for one user item pair, we make a full similarity matrix across all users, so that we can use this for a full recommendation engine. Other than that, it is pretty similar to the classroom implementation in terms of method. The second method we implemented is Time Series, of which we built 2. The first one simply represents sales over time for every single item in our dataset over time, and allows users to dim or highlight certain lines to view their trends. The second one is an ARIMA Time Series model using statsmodels’ ARIMA function in Python. We used this model in order to forecast future sales for particular items based on a combination of AR(p) and MA(q) frameworks, eventually settling on the one that gave us the least error scores.
    We wished to combine these two elements such that a manager at a retail company could look at similar items using the similarity matrix we constructed, take that information and cross reference it with the time series forecasting of said items in order to draw conclusions on which items would be great to sell together at what time based on whether their trends aligned. 
    """),

    html.H5("Data source: www.kaggle.com/datasets/fekihmea/fashion-retail-sales"),
    html.H5("References"),
    html.P("""
    Kao, Yi-Hsuan, and Hady W. Lauw. Knowledge-Enhanced Neural Fashion Trend Forecasting. Singapore Management University, 2023, ink.library.smu.edu.sg/cgi/viewcontent.cgi?params=/context/sis_research/article/8677/&path_info=Knowledge_enhanced_neural_fashion_trend_forecasting.pdf. Accessed 13 Apr. 2025.
    Wilson, Steve. Keeping Up With Today’s Retail Trends, MarketSource. https://www.marketsource.com/blog/keeping-up-todays-retail-trends/
    """),
    ], className="mt-4")


sales_layout = dbc.Container([
    html.H2("Seasonal Sales Trends"),
    dcc.Dropdown(
        id='sales-dropdown',
        options=[{'label': i, 'value': i} for i in sorted(df['item'].unique())],
        value=['Tunic', 'Jeans'],
        multi=True,
        placeholder='Select items'
    ),
    dcc.Graph(id='sales-graph')
], className="mt-4")

forecast_layout = dbc.Container([
    html.H2("Forecasting Sales"),
    html.Label("Select Item:"),
    dcc.Dropdown(
        id='forecast-dropdown',
        options=[{'label': i, 'value': i} for i in sorted(df['item'].unique())],
        value='Tunic'
    ),
    html.Label("Forecast Type:"),
    dcc.RadioItems(
        id='forecast-type',
        options=[
            {'label': 'Testing', 'value': 'backtest'},
            {'label': 'Future', 'value': 'future'}
        ],
        value='backtest',
        inline=True
    ),
    html.Label("Forecast Period (months):"),
    dcc.Slider(
        id='forecast-slider',
        min=3,
        max=24,
        step=1,
        value=6,
        marks={i: str(i) for i in range(3, 25, 3)}
    ),
    dcc.Graph(id='forecast-graph')
], className="mt-4")

similarity_layout = dbc.Container([
    html.H2("Item Similarity", className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=generate_similarity_heatmap(similarity_df)), md=7),
        dbc.Col([
            html.H5("Find Top 5 Similar Items"),
            dcc.Dropdown(
                id='similar-item-dropdown',
                options=[{'label': item, 'value': item} for item in similarity_df.columns],
                placeholder='Select item to find similar products',
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='similar-items-table')
        ], md=5)
    ])
], className="mt-4")

bundle_layout = dbc.Container([
    html.H2("Bundle Recommendations"),
    dcc.Dropdown(
        id='bundle-dropdown',
        options=[{'label': i, 'value': i} for i in bundle_df['item'].unique()],
        value='Tunic'
    ),
    dcc.Graph(id='bundle-graph')
], className="mt-4")

segmentation_layout = dbc.Container([
    html.H2("Sales by Payment & Review Level"),
    dcc.Dropdown(
        id='segmentation-dropdown',
        options=[{'label': i, 'value': i} for i in segmented_df['item'].unique()],
        value='Tunic'
    ),
    dcc.Graph(id='segmentation-graph')
], className="mt-4")

customer_layout = dbc.Container([
    html.H2("Customer Purchase History"),
    
    dcc.Dropdown(
        id='customer-dropdown',
        options=[{'label': i, 'value': i} for i in df['customerID'].unique()],
        value=df['customerID'].unique()[0],
        style={'marginBottom': '20px'}
    ),
    
    dbc.Row([
        dbc.Col(html.Div(id='customer-table'), md=6),
        dbc.Col(dcc.Graph(id='customer-review-graph'), md=6)
    ])
], className="mt-4")

timing_layout = dbc.Container([
    html.H2("Item Timing Table"),
    dcc.Graph(
        figure={
            'data': [{
                'type': 'table',
                'header': {'values': list(timing_df.columns)},
                'cells': {'values': [timing_df[col] for col in timing_df.columns]}
            }],
            'layout': {'height': 1000}
        }
    )
], className="mt-4")

# navigation layout
app.layout = dbc.Container([
    html.H1("Retail Intelligence Dashboard", className="text-center my-4"),
    html.H3("Forecasting Fashion: Item-Level Insights for Smarter Retail Decisions", className="text-center my-4"),
    html.P("By Vivian Li and Ryan Wu", className="text-center my-4"),
    dcc.Tabs(id='tabs', value='tab-home', children=[
        dcc.Tab(label='Home', value='tab-home'),
        dcc.Tab(label='Sales Trends', value='tab-sales'),
        dcc.Tab(label='Forecasting Sales', value='tab-forecast'),
        dcc.Tab(label='Similarity Items', value='tab-similarity'),
        dcc.Tab(label='Bundles Rec', value='tab-bundles'),
        dcc.Tab(label='Customers History', value='tab-customers'),
        dcc.Tab(label='Item Timing', value='tab-timing'),
        dcc.Tab(label='Sales Segmentation', value='tab-segmentation'),
    ], className='mb-4', persistence=True),
    html.Div(id='tabs-content')
], fluid=True)

# callback to render tab content
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-home':
        return home_layout
    elif tab == 'tab-sales':
        return sales_layout
    elif tab == 'tab-forecast':
        return forecast_layout
    elif tab == 'tab-similarity':
        return similarity_layout
    elif tab == 'tab-bundles':
        return bundle_layout
    elif tab == 'tab-segmentation':
        return segmentation_layout
    elif tab == 'tab-customers':
        return customer_layout
    elif tab == 'tab-timing':
        return timing_layout

# callbacks to update visualizations
@app.callback(Output('sales-graph', 'figure'), Input('sales-dropdown', 'value'))
def update_sales_trends(items):
    return plot_seasonal_sales_trends(monthly_sales, items)

@app.callback(Output('forecast-graph', 'figure'),
              Input('forecast-dropdown', 'value'),
              Input('forecast-type', 'value'),
              Input('forecast-slider', 'value'))
def update_forecast(item, forecast_type, period):
    if forecast_type == 'future':
        return forecast_item_future(monthly_sales, item, period)
    else:
        return forecast_item_backtest(monthly_sales, item)

@app.callback(Output('similar-items-table', 'figure'), Input('similar-item-dropdown', 'value'))
def update_similar_items(item_name):
    if not item_name:
        # return an empty layout 
        return {'data': [], 'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}}}
    
    df_top = get_top_similar_items(similarity_df, item_name)
    return {
        'data': [{
            'type': 'table',
            'header': {'values': list(df_top.columns)},
            'cells': {'values': [df_top[col] for col in df_top.columns]}
        }],
        'layout': {'height': 400}
    }

@app.callback(Output('bundle-graph', 'figure'), Input('bundle-dropdown', 'value'))
def update_bundle_chart(item):
    df_bundles = bundle_df[bundle_df['item'] == item]
    return {
        'data': [{
            'type': 'bar',
            'x': df_bundles['recommended_bundle'],
            'y': df_bundles['bundle_score'],
            'marker': {'color': 'teal'}
        }],
        'layout': {
            'title': f"Top Bundles for {item}",
            'xaxis': {'title': 'Recommended Item'},
            'yaxis': {'title': 'Bundle Score'},
            'template': 'plotly_white'
        }
    }

@app.callback(Output('segmentation-graph', 'figure'), Input('segmentation-dropdown', 'value'))
def update_segmentation(item):
    df_seg = segmented_df[segmented_df['item'] == item]
    return {
        'data': [{
            'type': 'bar',
            'x': df_seg['payment'],
            'y': df_seg['total_sales'],
            'text': df_seg['review_level'],
            'hovertemplate': 'Review: %{text}<br>Sales: $%{y:.2f}<extra></extra>',
            'marker': {'color': 'steelblue'}
        }],
        'layout': {
            'title': f"Sales by Payment & Review Level for {item}",
            'xaxis': {'title': 'Payment Method'},
            'yaxis': {'title': 'Total Sales'},
            'template': 'plotly_white'
        }
    }

@app.callback(Output('customer-table', 'children'),
              Output('customer-review-graph', 'figure'),
              Input('customer-dropdown', 'value'))
def update_customer_view(customer_id):
    history = get_customer_purchase_history(df, customer_id)
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in ['date', 'item', 'amount_usd', 'review', 'payment']])),
        html.Tbody([
            html.Tr([html.Td(row[col]) for col in ['date', 'item', 'amount_usd', 'review', 'payment']])
            for _, row in history.iterrows()
        ])
    ])
    review_fig = plot_customer_review_trend(df, customer_id)
    return table, review_fig

if __name__ == "__main__":
    app.run_server(debug=True)