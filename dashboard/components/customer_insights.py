import pandas as pd
import plotly.graph_objects as go

def get_customer_purchase_history(df, customer_id):
    # filter purchases for a specific customer and sort by date
    return df[df['customerID'] == customer_id].sort_values(by='date')

def plot_customer_review_trend(df, customer_id):
    # get and sort data for the selected customer
    cust_data = df[df['customerID'] == customer_id].sort_values('date')

    # initialize the plotly figure
    fig = go.Figure()
    
    # add a line plot showing review scores over time
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

    # update chart layout
    fig.update_layout(
        title=f"review ratings over time for customer {customer_id}",
        xaxis_title='purchase date',
        yaxis_title='review score',
        yaxis=dict(range=[0, 5.5]),  # keep y-axis within review score bounds
        template='plotly_white',
        height=400
    )

    return fig
