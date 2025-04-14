import pandas as pd

def generate_bundle_recommendations(df, similarity_df):
    # create binary matrix for co-purchases
    basket = df.groupby(['customerID', 'item']).size().unstack(fill_value=0)
    basket[basket > 0] = 1

    co_purchase = basket.T.dot(basket).fillna(0)

    bundle_recommendations = []

    for item in co_purchase.columns:
        try:
            combined_score = (
                (co_purchase[item] / co_purchase[item].max()) * 0.5 +
                (similarity_df[item] / similarity_df[item].max()) * 0.5
            )
            top_bundle = combined_score.drop(item).sort_values(ascending=False).head(3)
            for related_item, score in top_bundle.items():
                bundle_recommendations.append({
                    'item': item,
                    'recommended_bundle': related_item,
                    'bundle_score': round(score, 3)
                })
        except Exception:
            continue

    return pd.DataFrame(bundle_recommendations)
