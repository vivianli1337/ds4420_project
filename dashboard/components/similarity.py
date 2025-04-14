import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# item-item
def compute_item_similarity(user_item_matrix):
    # convert the dataframe to a numpy array
    item_mat = user_item_matrix.to_numpy()
    
    # calculate the mean rating for each item (ignoring nan values)
    item_means = np.nanmean(item_mat, axis=0)
    
    # center the matrix by subtracting the mean from each rating
    item_mat_centered = item_mat - item_means

    # prepare for similarity matrix construction
    item_names = user_item_matrix.columns
    n_items = item_mat_centered.shape[1]
    similarity_df = pd.DataFrame(index=item_names, columns=item_names, dtype=float)

    # calculate cosine similarity between all pairs of items
    for i in range(n_items):
        for j in range(n_items):
            vec_i = item_mat_centered[:, i]
            vec_j = item_mat_centered[:, j]
            
            # find indices where both items have a rating
            shared = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            
            # compute similarity only if there are at least two shared ratings
            if np.sum(shared) > 1:
                sim = cosine_similarity(vec_i[shared].reshape(1, -1), vec_j[shared].reshape(1, -1))[0, 0]
            else:
                sim = np.nan
            
            similarity_df.iloc[i, j] = sim

    return similarity_df

# heatmap
def generate_similarity_heatmap(similarity_df):
    sim_df = similarity_df.copy()
    sim_df.index.name = 'item1'
    sim_df = sim_df.reset_index().melt(id_vars='item1', var_name='item2', value_name='similarity')

    # create a heatmap using plotly
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

    # update layout for better readability
    fig.update_layout(
        xaxis_title='item',
        yaxis_title='item',
        xaxis_tickangle=90,
        height=700,
        template='plotly_white'
    )
    return fig

# find similar items
def get_top_similar_items(similarity_df, item_name, top_n=5):
    # convert input to lowercase for flexible matching
    item_name_lower = item_name.lower()
    
    # get all item names
    all_items = list(similarity_df.columns)
    
    # find the item matching the input (case insensitive)
    matched_items = [item for item in all_items if item.lower() == item_name_lower]

    # return empty dataframe if item not found
    if not matched_items:
        return pd.DataFrame(columns=['similar_item', 'similarity'])

    # get the most similar items, excluding the item itself
    matched_item = matched_items[0]
    similar_items = similarity_df[matched_item].drop(matched_item).sort_values(ascending=False).head(top_n)
    
    return pd.DataFrame({'similar_item': similar_items.index, 'similarity': similar_items.values})
