import plotly.graph_objects as go

# create plotly time analysis
def plot_seasonal_sales_trends(ts_sales_full, selected_items):
    fig = go.Figure()

    for item in selected_items:
        # filter the time series data for the current item
        item_data = ts_sales_full[ts_sales_full['item'] == item]
        
        # add a trace to the figure for this item
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

    # update overall layout of the chart
    fig.update_layout(
        title="seasonal sales trends",
        xaxis_title="month",
        yaxis_title="total sales (usd)",
        hovermode='closest',
        template='plotly_white',
        height=600,
        legend=dict(
            # disable interactive legend
            itemclick=False,  
            itemdoubleclick=False 
        ),
    )
    
    return fig
