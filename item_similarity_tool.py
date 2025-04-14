import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

CLOTHING_ITEMS = [
    'Tunic', 'Tank Top', 'Leggings', 'Onesie', 'Jacket', 'Trousers', 'Jeans',
    'Pajamas', 'Trench Coat', 'Poncho', 'Romper', 'T-shirt', 'Shorts',
    'Blazer', 'Hoodie', 'Sweater', 'Blouse', 'Swimsuit', 'Kimono', 'Cardigan',
    'Dress', 'Camisole', 'Flannel Shirt', 'Polo Shirt', 'Overalls', 'Coat',
    'Vest', 'Jumpsuit', 'Raincoat', 'Skirt', 'Pants'
]

def load_and_clean_data(csv_path="retail_sales.csv"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['amount_usd'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df = df[df['item'].isin(CLOTHING_ITEMS)].copy()
    return df

def compute_item_similarity(df):
    user_item_matrix = df.pivot_table(index='customerID', columns='item', values='review', aggfunc='mean')
    item_mat = user_item_matrix.to_numpy()
    item_means = np.nanmean(item_mat, axis=0)
    item_mat_centered = item_mat - item_means

    item_names = user_item_matrix.columns
    n_items = item_mat_centered.shape[1]
    item_similarity_df = pd.DataFrame(index=item_names, columns=item_names, dtype=float)

    for i in range(n_items):
        for j in range(n_items):
            vec_i = item_mat_centered[:, i]
            vec_j = item_mat_centered[:, j]
            shared = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            if np.sum(shared) > 1:
                sim = cosine_similarity(vec_i[shared].reshape(1, -1), vec_j[shared].reshape(1, -1))[0, 0]
            else:
                sim = np.nan
            item_similarity_df.iloc[i, j] = sim
    return item_similarity_df

def find_similar_items(similarity_df, item_name, top_n=5):
    item_name_lower = item_name.lower()
    all_items = list(similarity_df.columns)
    matched_items = [item for item in all_items if item.lower() == item_name_lower]

    if not matched_items:
        print(f"item '{item_name}' not found. please check the spelling or try another item.")
        return

    matched_item = matched_items[0]
    similar_items = similarity_df[matched_item].drop(matched_item).sort_values(ascending=False).head(top_n)

    print(f"\ntop {top_n} items similar to '{matched_item}':")
    for similar_item, score in similar_items.items():
        print(f"{similar_item}: similarity score = {score:.2f}")

if __name__ == "__main__":
    df = load_and_clean_data()
    similarity_df = compute_item_similarity(df)

    user_input = input("\nenter an item name to find similar products: ").strip()
    n = int(input("how many similar items do you want to see? (default 5): ") or 5)
    find_similar_items(similarity_df, user_input, top_n=n)
