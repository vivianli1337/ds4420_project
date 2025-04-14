# extra
import pandas as pd

def segment_sales_by_review_and_payment(df):
    # define review score bins and corresponding labels
    bins = [0, 2, 3.5, 5]
    labels = ['low (<=2)', 'medium (2-3.5)', 'high (>3.5)']
    
    # assign each row a review level based on review score
    df['review_level'] = pd.cut(df['review'], bins=bins, labels=labels, include_lowest=True)

    # group data by item, payment method, and review level
    segmented_sales = df.groupby(['item', 'payment', 'review_level']).agg(
        total_sales=('amount_usd', 'sum'),   # total sales amount
        units_sold=('item', 'count'),        # number of units sold
        avg_review=('review', 'mean')        # average review score
    ).reset_index().sort_values(['item', 'total_sales'], ascending=[True, False])

    return segmented_sales
