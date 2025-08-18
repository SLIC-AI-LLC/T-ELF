import os
from pathlib import Path
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd

from ...helpers.stats import H_clustering, top_words
from ...helpers.figures import create_wordcloud, plot_H_clustering
from ...helpers.file_system import check_path
from ...post_processing.Fox.post_process_functions import get_core_map, H_cluster_argmax, best_n_papers
from ...pre_processing.Vulture.tokens_analysis.top_words import get_top_words


class ClusteringAnalyzer:
    # ──────────────────────────────────────────────────────────────────
    def __init__(self, **kwargs):
        defaults = dict(
            top_n_words=50,
            out_dir='./post_result',
            archive_subdir='archive',
            default_clean_col='clean_title_abstract',
            wordcloud_size=(800, 800),
            max_font_size=80,
            contour_width=1,
            col_year='year',
            col_type='type',
            col_cluster='cluster',
            col_cluster_coords='cluster_coordinates',
            col_similarity='similarity_to_cluster_centroid',
            table_filename='table_H-clustering.csv',
            cluster_doc_map_filename='cluster_documents_map.txt',
            top_words_filename='top_words.csv',
            probs_filename='probabilities_top_words.csv',
            clusters_info_filename='clusters_information.csv',
            documents_info_filename='documents_information.csv'
        )
        self.__dict__.update({**defaults, **kwargs})

    # ──────────────────────────────────────────────────────────────────
    # public entry point (now supports hierarchical HNMFk as well)
    # ──────────────────────────────────────────────────────────────────
    def analyze(
        self,
        df,
        W: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        vocab: Optional[np.ndarray] = None,
        cluster_col: Optional[str] = None,
        *,
        hnmfk_model=None,
        clean_cols_name=None,
        process_parents=False,
        skip_completed=True,
    ):
        """
        Main dispatcher.

        • W/H/vocab present      → single NMF-k matrix
        • hnmfk_model + vocab    → hierarchical   (runs each node)
        • cluster_col provided   → pre-existing labels
        • else                   → pass-through (everything in one cluster)
        """
        self.archive_dir = self._ensure_dirs()
        df_copy = df.copy()
        self.default_clean_col = clean_cols_name or self.default_clean_col

        if hnmfk_model is not None and vocab is not None:
            return self.analyze_hnmfk(
                hnmfk_model, vocab, df_copy,
                clean_cols_name=clean_cols_name or self.default_clean_col,
                process_parents=process_parents,
                skip_completed=skip_completed,
            )

        if W is not None and H is not None and vocab is not None:
            return self._analyze_factor_model(df_copy, W, H, vocab)
        elif cluster_col and cluster_col in df_copy:
            return self._analyze_label_based(df_copy, cluster_col)
        else:
            return self._analyze_pass_through(df_copy)

    # ──────────────────────────────────────────────────────────────────
    # hierarchical HNMF-k processor
    # ──────────────────────────────────────────────────────────────────
    def analyze_hnmfk(
        self,
        hnmfk_model,
        vocab,
        data_df,
        *,
        clean_cols_name,
        process_parents=False,
        skip_completed=True,
    ) -> List[Path]:
        """
        Run post-processing on every leaf node (or all nodes if
        `process_parents=True`) of a fitted HNMFk model.

        Returns a list of CSV paths created.
        """
        vocab = np.asarray(vocab)
        out_csvs: List[Path] = []

        for node in hnmfk_model.traverse_nodes():
            if not (node["leaf"] or process_parents):
                continue

            # ── gather W / H for this node ────────────────────────────
            W = node.get("W")
            H = node.get("H")
            if W is None and H is None:  # signature-only node
                W = node["signature"].reshape(-1, 1)
                H = node["probabilities"].reshape(1, -1)

            node_dir = Path(node["node_save_path"]).resolve().parent
            csv_out = node_dir / f"cluster_for_k={W.shape[1]}.csv"
            if skip_completed and csv_out.exists():
                out_csvs.append(csv_out)
                continue

            idxs = list(node["original_indices"])
            node_df = data_df.iloc[idxs].reset_index(drop=True)

            # redirect outputs into the node’s folder
            prev_out_dir = self.out_dir
            self.out_dir = str(node_dir)
            self.archive_dir = self._ensure_dirs()

            self._analyze_factor_model(node_df, W, H, vocab)

            # restore global out_dir
            self.out_dir = prev_out_dir
            out_csvs.append(csv_out)

        return out_csvs

    # ──────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────
    def _ensure_dirs(self):
        archive_dir = os.path.join(self.out_dir, self.archive_subdir)
        os.makedirs(archive_dir, exist_ok=True)
        return archive_dir

    def _save_cluster_info(self, labels, table_df):
        table_df.to_csv(os.path.join(self.out_dir, self.table_filename), index=False)
        np.savetxt(os.path.join(self.archive_dir, self.cluster_doc_map_filename),
                   labels, fmt='%d')

    def _save_cluster_text_outputs(self, cluster_id, df, vocab=None, W=None):
        save_dir = check_path(os.path.join(self.out_dir, str(cluster_id)))
        # docs = df[self.default_clean_col].to_dict()
        docs = (
            df[self.default_clean_col]
            .fillna("")          # replace NaN with empty string
            .astype(str)         # ensure everything is a str
            .to_dict()
        )

        for n in (1, 2):
            bow = get_top_words(docs, top_n=100, n_gram=n, verbose=True)
            suffix = 'unigrams' if n == 1 else 'bigrams'
            bow.to_csv(os.path.join(save_dir, f'{cluster_id}_bow_{suffix}.csv'), index=False)

        if vocab is None or W is None:  # fallback: simple term-freq
            tokens = [tok for doc in docs.values() for tok in doc.split()]
            freq = Counter(tokens)
            vocab, W = list(freq.keys()), np.array(list(freq.values())).reshape(-1, 1)

        create_wordcloud(
            W=W,
            vocab=vocab,
            top_n=self.top_n_words,
            path=save_dir,
            max_words=self.top_n_words,
            mask=np.zeros(self.wordcloud_size),
            background_color='black',
            max_font_size=self.max_font_size,
            contour_width=self.contour_width
        )

    def _rank_docs(self, df):
        for col in (self.col_similarity, self.col_year, 'citations', 'references'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.nlargest(self.top_n_words, col)

        for col in ('author_ids', 'authors'):
            if col in df.columns:
                df = df.assign(author_count=df[col].apply(
                    lambda x: len(str(x).split(';')) if pd.notna(x) else 0))
                return df.nlargest(self.top_n_words, 'author_count')

        return df.sample(min(len(df), self.top_n_words), random_state=42)

    def _save_best_docs(self, df):
        for cid in df[self.col_cluster].dropna().unique():
            cdf = df[df[self.col_cluster] == cid].copy()
            best_df = self._rank_docs(cdf)
            out_dir = os.path.join(self.out_dir, str(int(cid)))
            best_df.to_csv(os.path.join(out_dir,
                                        f'best_{self.top_n_words}_docs_in_{cid}.csv'),
                           index=False)

    # ──────────────────────────────────────────────────────────────────
    # core analysis modes
    # ──────────────────────────────────────────────────────────────────
    def _analyze_factor_model(self, df, W, H, vocab):
        labels, counts, table_df = H_cluster_argmax(H)
        self._save_cluster_info(labels, table_df)

        vocab_arr = np.array(vocab)
        words, probs = top_words(W, vocab_arr, self.top_n_words)
        pd.DataFrame(words).to_csv(os.path.join(self.out_dir,
                                                self.top_words_filename),
                                   index=False)
        pd.DataFrame(probs).to_csv(os.path.join(self.out_dir,
                                                self.probs_filename),
                                   index=False)

        create_wordcloud(
            W=W,
            vocab=vocab_arr,
            top_n=self.top_n_words,
            path=self.out_dir,
            max_words=self.top_n_words,
            mask=np.zeros(self.wordcloud_size),
            background_color='black',
            max_font_size=self.max_font_size,
            contour_width=self.contour_width,
            grid_dimension=4
        )

        clusters_info, docs_info = H_clustering(H, verbose=True)
        pd.DataFrame(clusters_info).T.to_csv(os.path.join(
            self.archive_dir, self.clusters_info_filename), index=False)
        df_docs = pd.DataFrame(docs_info).T
        df_docs.to_csv(os.path.join(
            self.archive_dir, self.documents_info_filename), index=False)

        df[self.col_cluster] = df_docs['cluster']
        if 'cluster_coordinates' in df_docs:
            df[self.col_cluster_coords] = df_docs['cluster_coordinates']
        df[self.col_similarity] = df_docs['similarity_to_cluster_centroid']

        if self.col_year in df:
            df[self.col_year] = (df[self.col_year].fillna(-1)
                                 .astype(int).replace(-1, np.nan))
        if self.col_type in df:
            core_map = get_core_map(df)
            total_core = (df[self.col_type] == 0).sum()
            if total_core:
                table_df['core_count'] = table_df['cluster'].map(core_map)
                table_df['core_percentage'] = table_df['cluster'].map(
                    lambda c: round(100 * core_map.get(c, 0) / total_core, 2))
                table_df.to_csv(os.path.join(self.out_dir,
                                             self.table_filename), index=False)

        out_csv = f'cluster_for_k={W.shape[1]}.csv'
        df.to_csv(os.path.join(self.out_dir, out_csv), index=False)

        for cid in sorted(df[self.col_cluster].dropna().unique()):
            self._save_cluster_text_outputs(
                cid,
                df[df[self.col_cluster] == cid],
                vocab=vocab_arr,
                W=W[:, int(cid)].reshape(-1, 1)
            )
            plot_H_clustering(
                H[int(cid), :].reshape(1, -1),
                name=os.path.join(self.out_dir, str(cid),
                                  f'centroids_H_clustering_{cid}.png')
            )

        self._save_best_docs(df)
        return os.path.join(self.out_dir, out_csv)

    def _analyze_label_based(self, df, cluster_col):
        df[self.col_cluster] = df[cluster_col]
        labels = df[cluster_col].values
        counts = pd.Series(labels).value_counts()
        table_df = pd.DataFrame({"cluster": counts.index,
                                 "count": counts.values})

        self._save_cluster_info(labels, table_df)
        for cid in counts.index:
            self._save_cluster_text_outputs(cid, df[df[cluster_col] == cid])
        self._save_best_docs(df)

        out_csv = 'cluster_for_labels.csv'
        df.to_csv(os.path.join(self.out_dir, out_csv), index=False)
        return os.path.join(self.out_dir, out_csv)

    def _analyze_pass_through(self, df):
        df[self.col_cluster] = 0
        labels = df[self.col_cluster].values
        table_df = pd.DataFrame({"cluster": [0], "count": [len(df)]})

        self._save_cluster_info(labels, table_df)
        self._save_cluster_text_outputs(0, df)
        self._save_best_docs(df)

        out_csv = 'clustered_pass_through.csv'
        df.to_csv(os.path.join(self.out_dir, out_csv), index=False)
        return os.path.join(self.out_dir, out_csv)
