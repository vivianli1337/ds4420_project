from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def forecast_item_backtest(monthly_sales, item_name):
    item_data = monthly_sales[monthly_sales['item'] == item_name].copy()
    item_data = item_data.set_index('month').resample('M').sum().fillna(0)

    if len(item_data) < 12:
        return go.Figure().update_layout(title=f"Not enough data for {item_name}")

    split_idx = int(len(item_data) * 0.8)
    train = item_data.iloc[:split_idx]
    test = item_data.iloc[split_idx:]

    try:
        model = ARIMA(train['Total_Sales'], order=(0, 0, 1))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=len(test))
    except Exception as e:
        return go.Figure().update_layout(title=f"Forecast failed: {e}")

    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    forecast_index = test.index

    margin_error = ((conf_int.iloc[:, 1] - conf_int.iloc[:, 0]) / (2 * forecast_mean.abs())) * 100
    abs_error = (forecast_mean - test['Total_Sales']).abs()
    mape = (abs_error / test['Total_Sales'].replace(0, np.nan)).mean() * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['Total_Sales'], mode='lines+markers', name='Actual Sales'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
        fill='toself', name='95% CI', fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(x=forecast_index, y=margin_error, mode='markers', name='Error Margin (%)'))

    fig.update_layout(
        title=f"Backtest Forecast for {item_name} | MAPE: {mape:.2f}%",
        xaxis_title="Month", yaxis_title="Sales", template='plotly_white'
    )
    return fig

def forecast_item_future(monthly_sales, item_name, periods=6):
    item_data = monthly_sales[monthly_sales['item'] == item_name].copy()
    item_data = item_data.set_index('month').resample('M').sum().fillna(0)

    try:
        model = ARIMA(item_data['Total_Sales'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=periods)
    except Exception as e:
        return go.Figure().update_layout(title=f"Forecast failed: {e}")

    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    forecast_index = pd.date_range(start=item_data.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq='MS')

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['Total_Sales'], mode='lines+markers', name='Historical Sales'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
        fill='toself', name='95% CI', fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"{periods}-Month Forecast for {item_name}",
        xaxis_title="Month", yaxis_title="Sales", template='plotly_white'
    )
    return fig
