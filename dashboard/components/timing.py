# extra
import pandas as pd
# When should I market this product?
# Which items peak in summer vs. winter?

def calculate_item_timing(monthly_sales):
    # get the first month each item appeared in the dataset
    first_months = monthly_sales.groupby('item')['month'].min().reset_index(name='first_month')
    
    # get the month with highest sales (peak) for each item
    peak_months = monthly_sales.loc[
        monthly_sales.groupby('item')['Total_Sales'].idxmax()
    ][['item', 'month']].rename(columns={'month': 'peak_month'})

    # merge first and peak months into one dataframe
    timing_df = pd.merge(first_months, peak_months, on='item')
    
    # calculate the number of months between first and peak month (lead time)
    timing_df['lead_time_months'] = (
        timing_df['peak_month'].dt.to_period('M') - timing_df['first_month'].dt.to_period('M')
    ).apply(lambda x: x.n)

    return timing_df
