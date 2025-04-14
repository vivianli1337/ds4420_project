import pandas as pd

def segment_sales_by_review_and_payment(df):
    bins = [0, 2, 3.5, 5]
    labels = ['low (<=2)', 'medium (2-3.5)', 'high (>3.5)']
    df['review_level'] = pd.cut(df['review'], bins=bins, labels=labels, include_lowest=True)

    segmented_sales = df.groupby(['item', 'payment', 'review_level']).agg(
        total_sales=('amount_usd', 'sum'),
        units_sold=('item', 'count'),
        avg_review=('review', 'mean')
    ).reset_index().sort_values(['item', 'total_sales'], ascending=[True, False])

    return segmented_sales
