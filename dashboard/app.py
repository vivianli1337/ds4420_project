from dash import Dash, html, dcc, Input, Output
import pandas as pd

from data_loader import load_and_clean_data, prepare_monthly_sales
from components.similarity import compute_item_similarity, generate_similarity_heatmap, get_top_similar_items
from components.sales_trends import plot_seasonal_sales_trends
from components.forecast import forecast_item_backtest, forecast_item_future
from components.bundles import generate_bundle_recommendations
from components.segmentation import segment_sales_by_review_and_payment
from components.customer_insights import get_customer_purchase_history, plot_customer_review_trend
from components.timing import calculate_item_timing

# load data
df = load_and_clean_data()
monthly_sales = prepare_monthly_sales(df)
user_item_matrix = df.pivot_table(index='customerID', columns='item', values='review', aggfunc='mean')
similarity_df = compute_item_similarity(user_item_matrix)
bundle_df = generate_bundle_recommendations(df, similarity_df)
segmented_df = segment_sales_by_review_and_payment(df)
timing_df = calculate_item_timing(monthly_sales)

# app setup
app = Dash(__name__, external_stylesheets=['/assets/styles.css'])
server = app.server

app.layout = html.Div(className='container', children=[
    html.H1("🛍 Retail Dashboard"),

    html.Div(className='section', children=[
        html.H2("📈 Seasonal Sales Trends"),
        dcc.Dropdown(
            id='sales-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['item'].unique())],
            value=['Tunic', 'Jeans'],
            multi=True,
            placeholder='Select items'
        ),
        dcc.Graph(id='sales-graph')
    ]),

    html.Div(className='section', children=[
        html.H2("🔮 Forecasting"),
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
                {'label': 'Backtest', 'value': 'backtest'},
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
    ]),


    html.Div(className='section', children=[
        html.H2("📊 Item Similarity Heatmap"),
        dcc.Graph(figure=generate_similarity_heatmap(similarity_df))
    ]),

    html.Div(className='section', children=[
        html.H2("🧭 Find Similar Items"),
        dcc.Input(id='input-item', type='text', placeholder='Enter item name...'),
        dcc.Graph(id='similar-items-table')
    ]),

    html.Div(className='section', children=[
        html.H2("🤝 Bundle Recommendations"),
        dcc.Dropdown(
            id='bundle-dropdown',
            options=[{'label': i, 'value': i} for i in bundle_df['item'].unique()],
            value='Tunic',
            placeholder='Select item'
        ),
        dcc.Graph(id='bundle-graph')
    ]),

    html.Div(className='section', children=[
        html.H2("📊 Sales by Payment & Review Level"),
        dcc.Dropdown(
            id='segmentation-dropdown',
            options=[{'label': i, 'value': i} for i in segmented_df['item'].unique()],
            value='Tunic'
        ),
        dcc.Graph(id='segmentation-graph')
    ]),

    html.Div(className='section', children=[
        html.H2("🧍 Customer Purchase History"),
        dcc.Dropdown(
            id='customer-dropdown',
            options=[{'label': i, 'value': i} for i in df['customerID'].unique()],
            value=df['customerID'].unique()[0]
        ),
        html.Div(id='customer-table'),
        dcc.Graph(id='customer-review-graph')
    ]),

    html.Div(className='section', children=[
        html.H2("📅 Item Timing Table"),
        html.Div([
            dcc.Graph(
                figure={
                    'data': [{
                        'type': 'table',
                        'header': {'values': list(timing_df.columns)},
                        'cells': {'values': [timing_df[col] for col in timing_df.columns]}
                    }],
                    'layout': {'height': 500}
                }
            )
        ])
    ])
])

@app.callback(
    Output('sales-graph', 'figure'),
    Input('sales-dropdown', 'value')
)
def update_sales_trends(items):
    return plot_seasonal_sales_trends(monthly_sales, items)

@app.callback(
    Output('forecast-graph', 'figure'),
    Input('forecast-dropdown', 'value'),
    Input('forecast-type', 'value'),
    Input('forecast-slider', 'value')
)
def update_forecast(item, forecast_type, period):
    if forecast_type == 'future':
        return forecast_item_future(monthly_sales, item, period)
    else:
        return forecast_item_backtest(monthly_sales, item)

@app.callback(
    Output('similar-items-table', 'figure'),
    Input('input-item', 'value')
)
def update_similar_items(item_name):
    if not item_name:
        return {}
    df_top = get_top_similar_items(similarity_df, item_name)
    return {
        'data': [{
            'type': 'table',
            'header': {'values': list(df_top.columns)},
            'cells': {'values': [df_top[col] for col in df_top.columns]}
        }]
    }

@app.callback(
    Output('bundle-graph', 'figure'),
    Input('bundle-dropdown', 'value')
)
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

@app.callback(
    Output('segmentation-graph', 'figure'),
    Input('segmentation-dropdown', 'value')
)
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

@app.callback(
    Output('customer-table', 'children'),
    Output('customer-review-graph', 'figure'),
    Input('customer-dropdown', 'value')
)
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