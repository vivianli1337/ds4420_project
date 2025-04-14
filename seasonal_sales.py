import pandas as pd
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# clothing items to include
CLOTHING_ITEMS = [
    'Tunic', 'Tank Top', 'Leggings', 'Onesie', 'Jacket', 'Trousers', 'Jeans',
    'Pajamas', 'Trench Coat', 'Poncho', 'Romper', 'T-shirt', 'Shorts',
    'Blazer', 'Hoodie', 'Sweater', 'Blouse', 'Swimsuit', 'Kimono', 'Cardigan',
    'Dress', 'Camisole', 'Flannel Shirt', 'Polo Shirt', 'Overalls', 'Coat',
    'Vest', 'Jumpsuit', 'Raincoat', 'Skirt', 'Pants'
]

# load and clean data
def load_and_clean_data(csv_path="retail_sales.csv"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['amount_usd'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df = df[df['item'].isin(CLOTHING_ITEMS)].copy()
    return df

# aggregate data
def prepare_sales_data(df):
    ts_sales_full = df.groupby(['item', 'month']).agg(
        Total_Sales=('amount_usd', 'sum'),
        Units_Sold=('item', 'count')
    ).reset_index()
    return ts_sales_full

# plot selected items
def plot_multiple_items(ts_sales_full, selected_items):
    if not selected_items:
        print("⚠️ no items selected to plot.")
        return

    fig = go.Figure()
    for item in selected_items:
        item_data = ts_sales_full[ts_sales_full['item'] == item]
        fig.add_trace(go.Scatter(
            x=item_data['month'],
            y=item_data['Total_Sales'],
            mode='lines+markers',
            name=item,
            customdata=item_data[['Units_Sold']],
            hovertemplate=(
                f"<b>{item}</b><br>" +
                "month: %{x|%b %Y}<br>" +
                "sales: $%{y:.2f}<br>" +
                "units sold: %{customdata[0]}<extra></extra>"
            ),
            line=dict(width=2)
        ))

    fig.update_layout(
        title="seasonal sales trends (multiple items)",
        xaxis_title="month",
        yaxis_title="total sales (usd)",
        hovermode='closest',
        template='plotly_white',
        height=600
    )
    fig.show()

if __name__ == "__main__":
    df = load_and_clean_data()
    ts_sales_full = prepare_sales_data(df)

    # replace with Dash input later
    selected_items = ['Tunic', 'T-shirt', 'Jeans', 'Sweater']
    
    plot_multiple_items(ts_sales_full, selected_items)
