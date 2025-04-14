import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

def compute_item_similarity(user_item_matrix):
    item_mat = user_item_matrix.to_numpy()
    item_means = np.nanmean(item_mat, axis=0)
    item_mat_centered = item_mat - item_means

    item_names = user_item_matrix.columns
    n_items = item_mat_centered.shape[1]
    similarity_df = pd.DataFrame(index=item_names, columns=item_names, dtype=float)

    for i in range(n_items):
        for j in range(n_items):
            vec_i = item_mat_centered[:, i]
            vec_j = item_mat_centered[:, j]
            shared = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            if np.sum(shared) > 1:
                sim = cosine_similarity(vec_i[shared].reshape(1, -1), vec_j[shared].reshape(1, -1))[0, 0]
            else:
                sim = np.nan
            similarity_df.iloc[i, j] = sim

    return similarity_df

def generate_similarity_heatmap(similarity_df):
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
        height=700,
        template='plotly_white'
    )
    return fig

def get_top_similar_items(similarity_df, item_name, top_n=5):
    item_name_lower = item_name.lower()
    all_items = list(similarity_df.columns)
    matched_items = [item for item in all_items if item.lower() == item_name_lower]

    if not matched_items:
        return pd.DataFrame(columns=['similar_item', 'similarity'])

    matched_item = matched_items[0]
    similar_items = similarity_df[matched_item].drop(matched_item).sort_values(ascending=False).head(top_n)
    return pd.DataFrame({'similar_item': similar_items.index, 'similarity': similar_items.values})
