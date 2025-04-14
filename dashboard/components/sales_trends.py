import plotly.graph_objects as go

def plot_seasonal_sales_trends(ts_sales_full, selected_items):
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
        line=dict(width=2),
        legendgroup=item,
        showlegend=True,
        visible=True
    ))

    fig.update_layout(
        title="seasonal sales trends (multiple items)",
        xaxis_title="month",
        yaxis_title="total sales (usd)",
        hovermode='closest',
        template='plotly_white',
        height=600,
        legend=dict(
        itemclick=False,
        itemdoubleclick=False
        ),
    )
    return fig
