import pandas as pd

def calculate_item_timing(monthly_sales):
    first_months = monthly_sales.groupby('item')['month'].min().reset_index(name='first_month')
    peak_months = monthly_sales.loc[
        monthly_sales.groupby('item')['Total_Sales'].idxmax()
    ][['item', 'month']].rename(columns={'month': 'peak_month'})

    timing_df = pd.merge(first_months, peak_months, on='item')
    timing_df['lead_time_months'] = (
        timing_df['peak_month'].dt.to_period('M') - timing_df['first_month'].dt.to_period('M')
    ).apply(lambda x: x.n)

    return timing_df
