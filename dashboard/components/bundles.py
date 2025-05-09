import pandas as pd

def generate_bundle_recommendations(df, similarity_df):
    # create a binary matrix indicating whether a customer bought an item (1) or not (0)
    basket = df.groupby(['customerID', 'item']).size().unstack(fill_value=0)
    basket[basket > 0] = 1

    # compute co-purchase frequency count by dot product of the binary basket matrix
    co_purchase = basket.T.dot(basket).fillna(0)

    # initialize list to store bundle recommendations
    bundle_recommendations = []

    # loop through each item to generate top bundle recommendations
    for item in co_purchase.columns:
        try:
            # combine co-purchase frequency and item similarity (equal weight)
            combined_score = (
                (co_purchase[item] / co_purchase[item].max()) * 0.5 +
                (similarity_df[item] / similarity_df[item].max()) * 0.5
            )

            # get top 3 recommended bundle items (excluding the item itself)
            top_bundle = combined_score.drop(item).sort_values(ascending=False).head(3)

            # store results
            for related_item, score in top_bundle.items():
                bundle_recommendations.append({
                    'item': item,
                    'recommended_bundle': related_item,
                    'bundle_score': round(score, 3)
                })
        except Exception:
            # skip items that fail due to missing data
            continue

    return pd.DataFrame(bundle_recommendations)
