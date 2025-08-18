import os
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore', category=UserWarning)
from ...helpers.file_system import check_path
from ...helpers.frames import calculate_term_representations

# constants
CONFIG_PATH = os.path.join("input", "config.json")

def get_core_map(df):
    core_map = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df.loc[df['cluster'] == cluster_id]
        core_map[cluster_id] = len(cluster_df.loc[cluster_df.type == 0])
    return core_map
    
def H_cluster_argmax(H):
    indices, labels, counts = np.unique(np.argmax(H,axis=0), return_counts=True, return_inverse=True)
    arr   = counts/np.sum(counts)
    table = pd.DataFrame({'cluster': indices, 'super_topic':indices, 'counts': counts, 'percentage':np.round(100*arr, 2)})
    return labels, counts, table


def best_n_papers(df, path, n):
    """
    Get the top n papers that are closest to the cluster centroid

    Parameters:
    -----------
    df: pd.DataFrame
        The processed DataFrame that contains a `cluster` and `similarity_to_cluster_centroid` columns
    path: str
        Where to save the best papers
    n: int
        How many papers to save
    """
    for cluster_id in sorted(df['cluster'].dropna().unique()):
        save_dir = check_path(os.path.join(path, f'{int(cluster_id)}'))
        cluster_df = df.loc[df['cluster'] == cluster_id].copy()
        cluster_df['similarity_to_cluster_centroid'] = pd.to_numeric(cluster_df['similarity_to_cluster_centroid'], errors='coerce')
        best_df = cluster_df.nlargest(n, ['similarity_to_cluster_centroid'])
        best_df.to_csv(os.path.join(save_dir, f'best_{n}_docs_in_{cluster_id}.csv'), index=False)


def sme_attribution(df, path, terms, col='clean_title_abstract'):
    for cluster_id in sorted(df['cluster'].unique()):
        save_dir = check_path(os.path.join(path, f'{int(cluster_id)}'))
        cluster_df = df.loc[df['cluster'] == cluster_id].copy() 
        attribution_df = calculate_term_representations(cluster_df, terms, col)
        attribution_df.to_csv(os.path.join(save_dir, f'{cluster_id}_attribution.csv'), index=False)
