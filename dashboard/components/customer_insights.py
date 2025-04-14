import pandas as pd
import plotly.graph_objects as go

def get_customer_purchase_history(df, customer_id):
    return df[df['customerID'] == customer_id].sort_values(by='date')

def plot_customer_review_trend(df, customer_id):
    cust_data = df[df['customerID'] == customer_id].sort_values('date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cust_data['date'],
        y=cust_data['review'],
        mode='lines+markers',
        name=f"customer {customer_id}",
        text=[f"item: {item}<br>amount: ${amt:.2f}" 
              for item, amt in zip(cust_data['item'], cust_data['amount_usd'])],
        hoverinfo='text+x+y',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title=f"review ratings over time for customer {customer_id}",
        xaxis_title='purchase date',
        yaxis_title='review score',
        yaxis=dict(range=[0, 5.5]),
        template='plotly_white',
        height=400
    )

    return fig
