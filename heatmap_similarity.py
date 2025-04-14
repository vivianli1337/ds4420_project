import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px

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

def plot_similarity_heatmap(similarity_df):
    sim_df = similarity_df.copy()
    sim_df.index.name = 'item1'
    sim_df = sim_df.reset_index().melt(id_vars='item1', var_name='item2', value_name='similarity')

    fig = px.density_heatmap(
        sim_df,
        x='item2',
        y='item1',
        z='similarity',
        color_continuous_scale='RdBu_r',
        hover_data={
            'item1': True,
            'item2': True,
            'similarity': ':.2f'
        },
        title='item-item similarity heatmap (cosine similarity)'
    )

    fig.update_layout(
        xaxis_title='item',
        yaxis_title='item',
        xaxis_tickangle=90,
        autosize=False,
        width=900,
        height=800
    )

    fig.show()

if __name__ == "__main__":
    df = load_and_clean_data()
    similarity_df = compute_item_similarity(df)
    plot_similarity_heatmap(similarity_df)