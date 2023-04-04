from typing import Dict
import pandas as pd


def preprocessor(df: Dict) -> Dict:
    user_ids = df['user_id'].unique()
    item_ids = df['movie_id'].unique()

    ratings_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')

    implicit_df = {}
    implicit_df['user_id'] = []
    implicit_df['movie_id'] = []
    implicit_df['implicit_feedback'] = []

    user_dict = {}
    item_dict = {}
    for u, user_id in enumerate(user_ids):
        user_dict[u] = user_id
        for i, item_id in enumerate(item_ids):
            if i not in item_dict:
                item_dict[i] = item_id
            implicit_df['user_id'].append(u)
            implicit_df['movie_id'].append(i)
            if pd.isna(ratings_matrix.loc[user_id, item_id]):
                implicit_df['implicit_feedback'].append(0)
            else:
                implicit_df['implicit_feedback'].append(1)

    implicit_df = pd.DataFrame(implicit_df)
    implicit_df['user_id'] = implicit_df['user_id'].astype('category')
    implicit_df['movie_id'] = implicit_df['movie_id'].astype('category')
    
    return implicit_df
