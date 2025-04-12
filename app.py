from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load the retail sales data
def load_data():
    return pd.read_csv('retail_sales.csv')

def process_dashboard_data():
    df = load_data()
    
    # Calculate summary metrics
    total_sales = df['amount_usd'].sum()
    total_customers = df['customerID'].nunique()
    avg_rating = df['review'].mean()
    total_items = len(df)
    
    # Sales over time
    sales_over_time = df.groupby('date')['amount_usd'].sum().reset_index()
    sales_chart = px.line(sales_over_time, x='date', y='amount_usd',
                         title='Sales Over Time')
    
    # Create line plot figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sales_over_time['date'],
        y=sales_over_time['amount_usd'],
        mode='lines+markers',
        name='Sales'
    ))
    fig.update_layout(
        title='Sales Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Sales Amount (USD)',
        template='plotly_white'
    )
    line_plot = fig.to_json()
    
    # Create time series plot by item
    df['month'] = pd.to_datetime(df['date']).dt.to_period("M")
    ts_sales = df.groupby(['item', 'month']).agg(
        Total_Sales=('amount_usd', 'sum')
    ).reset_index()
    ts_sales['month'] = ts_sales['month'].dt.to_timestamp()

    # Create time series plot by item
    ts_fig = go.Figure()

    for item in ts_sales['item'].unique():
        item_data = ts_sales[ts_sales['item'] == item]
        ts_fig.add_trace(go.Scatter(
            x=item_data['month'],
            y=item_data['Total_Sales'],
            mode='lines+markers',
            name=item,
            hoverinfo='text',
            text=[f"Item: {item}<br>Sales: ${val:,.2f}" for val in item_data['Total_Sales']],
            line=dict(width=2),
            opacity=0.6
        ))

    ts_fig.update_layout(
        title="Total Sales Over Time by Item",
        xaxis_title="Month",
        yaxis_title="Total Sales (USD)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    ts_plot = ts_fig.to_json()
    
    # Create time series plot
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and calculate metrics
    daily_metrics = df.groupby('date').agg({
        'amount_usd': 'sum',
        'customerID': 'nunique',
        'item': 'count'
    }).reset_index()
    
    daily_metrics.columns = ['date', 'total_sales', 'unique_customers', 'items_sold']
    
    # Create time series plot
    time_series_fig = go.Figure()
    
    # Add traces for each metric
    time_series_fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['total_sales'],
        mode='lines+markers',
        name='Total Sales',
        line=dict(color='blue')
    ))
    
    time_series_fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['unique_customers'],
        mode='lines+markers',
        name='Unique Customers',
        line=dict(color='green')
    ))
    
    time_series_fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['items_sold'],
        mode='lines+markers',
        name='Items Sold',
        line=dict(color='red')
    ))
    
    # Update layout
    time_series_fig.update_layout(
        title='Daily Sales Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Count',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    time_series_plot = time_series_fig.to_json()
    
    # Interactive item sales chart
    item_counts = df['item'].value_counts().reset_index()
    item_counts.columns = ['item', 'count']
    
    item_sales_fig = go.Figure()
    item_sales_fig.add_trace(go.Bar(
        x=item_counts['item'],
        y=item_counts['count'],
        name='Items Sold',
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>' +
                     'Items Sold: %{y}<br>' +
                     '<extra></extra>'
    ))
    
    item_sales_fig.update_layout(
        title='Interactive Item Sales Analysis',
        xaxis=dict(
            title='Product',
            tickangle=-45,
            tickmode='array',
            ticktext=item_counts['item'],
            tickvals=item_counts['item']
        ),
        yaxis=dict(
            title='Number of Items Sold'
        ),
        height=600,
        template='plotly_white',
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
                buttons=list([
                    dict(
                        label="All Items",
                        method="update",
                        args=[{"visible": [True]}]
                    ),
                    dict(
                        label="Top 10",
                        method="update",
                        args=[{"visible": [True]}, {"x": [item_counts['item'][:10]]}]
                    ),
                    dict(
                        label="Bottom 10",
                        method="update",
                        args=[{"visible": [True]}, {"x": [item_counts['item'][-10:]]}]
                    )
                ])
            )
        ]
    )
    item_sales_fig.update_xaxes(rangeslider_visible=True)
    item_sales_chart = item_sales_fig.to_json()
    
    # Top selling items
    top_items = df.groupby('item')['amount_usd'].sum().sort_values(ascending=False).head(10)
    top_items_chart = px.bar(top_items, title='Top Selling Items')
    
    # Customer similarity matrix
    customer_matrix = pd.pivot_table(
        df,
        values='amount_usd',
        index='customerID',
        columns='item',
        fill_value=0
    )
    similarity_matrix = cosine_similarity(customer_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=customer_matrix.index,
        columns=customer_matrix.index
    )
    
    # Item similarity matrix
    item_matrix = pd.pivot_table(
        df,
        values='amount_usd',
        index='item',
        columns='customerID',
        fill_value=0
    )
    item_similarity_matrix = cosine_similarity(item_matrix)
    item_similarity_df = pd.DataFrame(
        item_similarity_matrix,
        index=item_matrix.index,
        columns=item_matrix.index
    )
    
    # Payment distribution
    payment_dist = df['payment'].value_counts()
    payment_chart = px.pie(values=payment_dist.values, names=payment_dist.index,
                          title='Payment Method Distribution')
    
    # Convert DataFrames to JSON-serializable format
    similarity_matrix_json = similarity_df.to_dict()
    item_similarity_matrix_json = item_similarity_df.to_dict()
    
    return {
        'total_sales': f"{total_sales:,.2f}",
        'total_customers': total_customers,
        'avg_rating': f"{avg_rating:.1f}",
        'total_items': total_items,
        'sales_chart': sales_chart.to_json(),
        'line_plot': line_plot,
        'time_series_plot': time_series_plot,
        'ts_plot': ts_plot,
        'item_sales_chart': item_sales_chart,
        'top_items_chart': top_items_chart.to_json(),
        'similarity_matrix': json.dumps(similarity_matrix_json),
        'item_similarity_matrix': json.dumps(item_similarity_matrix_json),
        'payment_chart': payment_chart.to_json()
    }

@app.route('/')
def dashboard():
    data = process_dashboard_data()
    return render_template('dashboard.html', **data)

@app.route('/api/process-data', methods=['POST'])
def process_data():
    try:
        # Get the data from the request
        data = request.get_json()
        
        # Load the existing data
        df = load_data()
        
        # Process the data (you can add your specific processing logic here)
        # For example, calculating customer similarities
        customer_matrix = pd.pivot_table(
            df,
            values='amount_usd',
            index='customerID',
            columns='item',
            fill_value=0
        )
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(customer_matrix)
        
        # Convert to DataFrame for easier handling
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=customer_matrix.index,
            columns=customer_matrix.index
        )
        
        # Prepare the response data
        response_data = {
            'message': 'Data processed successfully',
            'similarity_matrix': similarity_df.to_dict(),
            'processed_timestamp': datetime.now().isoformat()
        }
        
        # Here you would typically send this data to Ull
        # For now, we'll just return it in the response
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 