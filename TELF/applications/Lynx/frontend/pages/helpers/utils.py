import streamlit as st
import subprocess
import platform
import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from collections import defaultdict
import re 
import networkx as nx
from pykeen import predict

def build_coauthorship_network(authors, author_ids):
    """
    Build a co-authorship network graph from the dataframe.
    Nodes represent authors, edges represent co-authorship occurrences.
    """
    G = nx.Graph()
    author_pairs = {}

    for authors, ids in zip(authors, author_ids):
        authors_list = preprocess_authors([authors], [ids])

        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                pair = tuple(sorted([authors_list[i], authors_list[j]]))
                if pair in author_pairs:
                    author_pairs[pair] += 1
                else:
                    author_pairs[pair] = 1

    for (author1, author2), count in author_pairs.items():
        G.add_edge(author1, author2, weight=count)

    return G

@st.cache_data
def get_top_words(documents,
                  top_n=10,
                  n_gram=1,
                  verbose=True,
                  filename=None) -> pd.DataFrame:
    """
    Collects statistics for the top words or n-grams. Returns a table with columns
    word, tf, df, df_fraction, and tf_fraction.

    - word column lists the words in the top_n.
    - tf is the term-frequency, how many times given word occured in documents.
    - df is the document-frequency, in how documents given word occured.
    - df_fraction is df / len(documents)
    - tf_fraction is tf / (total number of unique tokens or n-grams)

    Parameters
    ----------
    documents : list or dict
        list or dictionary of documents. 
        If dictionary, keys are the document IDs, values are the text.
    top_n : int, optional
        Top n words or n-grams to report. The default is 10.
    n_gram : int, optional
        1 is words, or n-grams when > 1. The default is 1.
    verbose : bool, optional
        Verbosity flag. The default is True.
    filename : str, optional
        If not one, saves the table to the given location.

    Returns
    -------
    pd.DataFrame
        Table for the statistics.

    """

    if isinstance(documents, dict):
        documents = list(documents.values())

    word_stats = defaultdict(lambda: {"tf": 0, "df": 0})

    for doc in tqdm(documents, disable=not verbose):
        tokens = doc.split()
        ngrams = zip(*[tokens[i:] for i in range(n_gram)])
        ngrams = [" ".join(ngram) for ngram in ngrams]

        for gram in ngrams:
            word_stats[gram]["tf"] += 1
        for gram in set(ngrams):
            word_stats[gram]["df"] += 1

    word_stats = dict(word_stats)
    top_words = dict(sorted(word_stats.items(), key=lambda x: x[1]["tf"], reverse=True)[:top_n])
    target_data = {"word": [], "tf": [], "df": [], "df_fraction": [], "tf_fraction": []}

    for word in top_words:
        target_data["word"].append(word)
        target_data["tf"].append(word_stats[word]["tf"])
        target_data["df"].append(word_stats[word]["df"])
        target_data["df_fraction"].append(word_stats[word]["df"] / len(documents))
        target_data["tf_fraction"].append(word_stats[word]["tf"] / len(word_stats))

    # put together the results
    table = pd.DataFrame.from_dict(target_data)

    if filename:
        table.to_csv(filename+".csv", index=False)

    return table



def extract_unique_values(data_key, data):
    """Extract unique values from nested lists in session state."""
    return list(set(a for alist in data[data_key].values() for a in alist))

def extract_all_affiliation_info(affiliations):
    organizations = list()
    countries = list()

    for curr_affil in affiliations:
        if not curr_affil or curr_affil in ["Nan", "nan", None, np.nan]:
            continue  # Skip invalid rows

        try:
            curr_affil = ast.literal_eval(curr_affil)  # Convert string to dictionary
            if not isinstance(curr_affil, dict):
                continue  # Ensure it's a dictionary

            for affil_id, values in curr_affil.items():
                if not isinstance(values, dict):
                    continue  # Ensure values is a dictionary

                # Extract and clean name
                name = values.get("name", "").strip()
                if not name or name in ["Nan", "nan", None, np.nan]:
                    continue  # Skip empty names

                # Extract and clean country
                country = values.get("country", "").strip()
                if country and country not in ["Nan", "nan", None, np.nan]:
                    countries.append(country)

                organizations.append(f"{name} {affil_id}")

        except (ValueError, SyntaxError):
            continue  # Handle cases where eval fails (e.g., malformed data)

    return list(organizations), list(countries)

def get_token_index_map(texts, tokens):
    index_map = {}

    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            continue  # Skip non-string entries
        for token in tokens:
            if token in text:
                index_map.setdefault(token, []).append(idx)

    return index_map

def extract_affiliation_info(affiliations, use_index_map=False):
    organizations = set()
    countries = set()
    org_index_map = {}
    country_index_map = {}

    for idx, curr_affil in enumerate(affiliations):
        if not curr_affil or curr_affil in ["Nan", "nan", None, np.nan]:
            continue  # Skip invalid rows

        try:
            curr_affil = ast.literal_eval(curr_affil)  # Convert string to dictionary
            if not isinstance(curr_affil, dict):
                continue  # Ensure it's a dictionary

            for affil_id, values in curr_affil.items():
                if not isinstance(values, dict):
                    continue  # Ensure values is a dictionary

                # Extract and clean name
                name = values.get("name", "").strip()
                if not name or name in ["Nan", "nan", None, np.nan]:
                    continue  # Skip empty names

                org_key = f"{name} {affil_id}"
                organizations.add(org_key)
                if use_index_map:
                    org_index_map.setdefault(org_key, []).append(idx)

                # Extract and clean country
                country = values.get("country", "").strip()
                if country and country not in ["Nan", "nan", None, np.nan]:
                    countries.add(country)
                    if use_index_map:
                        country_index_map.setdefault(country, []).append(idx)

        except (ValueError, SyntaxError):
            continue  # Handle cases where eval fails (e.g., malformed data)

    if use_index_map:
        return list(organizations), list(countries), org_index_map, country_index_map
    else:
        return list(organizations), list(countries)


def filter_documents(
    data_map,
    top_n,
    selected_words,
    token_search_type,
    attr_selections,
    search_categories,
    keywords_search_type,
    use_index_map=False,
    negative_words=None,
):
    """
    Returns:
        selected_labels               – {doc_label: 1}
        selected_indices (optional)   – {doc_label: sorted(list_of_indices)}
    """

    selected_labels  = {}
    selected_indices = {}        # populated only when use_index_map=True

    # ------------------------------------------------------------
    # Decide which unigram list is being searched
    keywords_search_field = (
        "unigrams" if keywords_search_type == "Keywords" else "denovo_unigrams"
    )
    # NOTE: we *always* use denovo_unigrams_index_map for index positions
    UNIGRAM_INDEX_MAP_FIELD = "denovo_unigrams_index_map"

    # ------------------------------------------------------------
    for doc_id, doc_val in data_map.items():

        # ============== 1) KEYWORD-BASED FILTER =================
        keyword_pass = True
        keyword_idx_set = set()

        if selected_words or negative_words:                       # only check if user gave tokens
            # Limit to Top-N if needed
            if top_n == "All":
                doc_unigrams = set(doc_val[keywords_search_field])
                max_allowed_idx = None
            else:
                doc_unigrams   = set(doc_val[keywords_search_field][:top_n])
                max_allowed_idx = top_n - 1

            # AND / OR logic
            if token_search_type == "and":
                keyword_pass = all(w in doc_unigrams for w in selected_words)
            else:
                keyword_pass = any(w in doc_unigrams for w in selected_words)
            
            if negative_words:  # Check if there are words to exclude
                keyword_pass = keyword_pass and not any(w in doc_unigrams for w in negative_words)

            if not keyword_pass:
                continue

            # Collect indices *only* for DeNovo searches
            if use_index_map and keywords_search_type != "Keywords":
                unigram_map = doc_val.get(UNIGRAM_INDEX_MAP_FIELD, {})

                # Collect indices of all negative words (if any)
                negative_idx_set = set()
                if negative_words:
                    for neg_word in negative_words:
                        negative_idx_set.update(unigram_map.get(neg_word, []))
                print(negative_idx_set, unigram_map)
                for w in selected_words:
                    for i in unigram_map.get(w, []):
                        if (max_allowed_idx is None or i <= max_allowed_idx) and i not in negative_idx_set:
                            keyword_idx_set.add(i)

        # ============== 2) ATTRIBUTE-BASED FILTERS ==============
        attr_pass = True
        # We’ll build one union set per attribute category; those unions
        # are later intersected across categories (and with keywords).
        per_category_sets = []

        for label, info in search_categories.items():
            attr_key      = info["key"]                # e.g. 'country'
            chosen_vals   = attr_selections.get(f"selected_{attr_key}", [])
            logic         = attr_selections.get(f"{attr_key}_search_type", "and")

            if not chosen_vals:
                continue                                # no filter for this attr

            doc_attr_set = set(doc_val.get(attr_key, []))

            if logic == "and":
                if not set(chosen_vals).issubset(doc_attr_set):
                    attr_pass = False
                    break
            else:
                if not doc_attr_set.intersection(chosen_vals):
                    attr_pass = False
                    break

            # Add index positions for this attribute category
            if use_index_map:
                attr_map = doc_val.get(f"{attr_key}_index_map", {})

                if logic == "and":
                    # intersection across every selected value
                    idx_set = None
                    for v in chosen_vals:
                        v_idx = set(attr_map.get(v, []))
                        idx_set = v_idx if idx_set is None else idx_set & v_idx
                else:  # "or"
                    # union across every selected value
                    idx_set = set()
                    for v in chosen_vals:
                        idx_set.update(attr_map.get(v, []))

                per_category_sets.append(idx_set)

        if not attr_pass:
            continue

        # ============== 3) CONSOLIDATE INDICES ==================
        if use_index_map:
            # Start with keyword indices (may be empty)
            if keyword_idx_set:
                consolidated = keyword_idx_set.copy()
            else:
                consolidated = None                     # will adopt first set

            # Intersect with every attribute category’s index set
            for s in per_category_sets:
                if consolidated is None:
                    consolidated = s.copy()
                else:
                    consolidated &= s

            # If we never gathered any sets, consolidated stays None
            final_idx_list = sorted(consolidated) if consolidated else None
            selected_indices[doc_val["label"]] = final_idx_list

        # ============== 4) RECORD DOCUMENT ======================
        selected_labels[doc_val["label"]] = 1

    # ------------------------------------------------------------
    if use_index_map:
        return selected_labels, selected_indices
    
    return selected_labels, None


def preprocess_authors(authors_col, author_ids_col):
    """
    Process author and author_id columns to ensure correct matching.
    Handles None, NaN, and "None" values.
    """
    processed_authors = []
    
    for authors, ids in zip(authors_col, author_ids_col):
        if pd.isna(authors) or authors in ["None", "none", ""]:
            authors_list = []
        else:
            authors_list = authors.split(";")

        if pd.isna(ids) or ids in ["None", "none", ""]:
            ids_list = []
        else:
            ids_list = ids.split(";")
        
        matched_authors = [
            f"{name.strip()} ({id_.strip()})" if name.strip() else f"( {id_.strip()} )"
            for name, id_ in zip(authors_list, ids_list)
        ]
        processed_authors.extend(matched_authors)
    
    return processed_authors

def split_and_flatten_list(lst, split_by=";", use_index_map=False):
    result = []
    index_map = {}

    for idx, item in enumerate(lst):
        if isinstance(item, float) and np.isnan(item):  # Handling NaN (float type)
            continue
        elif isinstance(item, str) and item.lower() == "nan":  # Handling "nan" as a string
            continue
        else:
            parts = [x.strip() for x in str(item).split(split_by) if x.strip()]
            result.extend(parts)
            if use_index_map:
                for part in parts:
                    index_map.setdefault(part, []).append(idx)

    if use_index_map:
        return result, index_map
    else:
        return result

def find_unique_special_chars(column):
    """
    Finds all unique special characters in a given pandas DataFrame column.
    
    :param column: A pandas Series (column) containing text data
    :return: A set of unique special characters found in the column
    """
    special_chars = set()
    
    # Regular expression to match special characters (excluding letters, numbers, and spaces)
    pattern = re.compile(r'[^a-zA-Z0-9\s]')
    
    # Iterate through the column
    for text in column.dropna():  # Drop NaN values to avoid errors
        special_chars.update(pattern.findall(str(text)))  # Extract and add unique special chars
    
    return special_chars

def load_csv_file_items(path, suffix, ends, column):
    file = find_by_suffix(directory=path, suffix=suffix, ends=ends)        
    if file:
        if isinstance(column, list):
            items = np.array(pd.read_csv(file)[column])
        else:
            items = np.array(pd.read_csv(file)[column].tolist())
    else:
        items = np.array([])
    return items

def find_files_by_extensions(directory, extensions=("html", "png")):
    files = {ext: [] for ext in extensions}
    if os.path.exists(directory) and os.path.isdir(directory):
        for file in os.listdir(directory):
            for ext in extensions:
                if file.endswith(f".{ext}"):
                    files[ext].append(os.path.join(directory, file))
    return files

def file_exists_in_path(path: str, filename: str) -> bool:
    """
    Checks if a file with the given filename exists in the specified path.

    :param path: The directory path to search in.
    :param filename: The name of the file to look for.
    :return: True if the file exists, False otherwise.
    """
    for root, _, files in os.walk(path):
        if filename in files:
            return True
    return False

def find_folder_by_prefix(directory, prefix):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            return item  # Returns the first matching folder found
    return None  # Returns None if no folder matches

def find_by_suffix(directory, suffix="_bow_bigrams.csv", ends=True):
    if os.path.exists(directory) and os.path.isdir(directory):
        for file in os.listdir(directory):
            if (ends and file.endswith(suffix)) or (not ends and file.startswith(suffix)):
                return os.path.join(directory, file)
    return None

@st.cache_data
def extract_attribute_lists(attributes_dict):
    organized_attributes = {}
    for node_name, attributes in attributes_dict.items():
        for attribute_type, attribute in attributes.items():
            if attribute_type not in organized_attributes:
                organized_attributes[attribute_type] = [attribute]
            else:
                organized_attributes[attribute_type].append(attribute)
    return organized_attributes

@st.cache_data
def extract_link_attributes_df(rows, cols, attributes_dict):
    all_unique_nodes = list(set(rows + cols))
    df_dict = {"node":[]}
    unique_attribute_types = []
    # collect all potential attributes
    for _, attributes in attributes_dict.items():
        for attribute_type, _ in attributes.items():
            if attribute_type not in df_dict:
                df_dict[attribute_type] = []
                unique_attribute_types.append(attribute_type)

    # form dataframe
    for node in all_unique_nodes:
        df_dict["node"].append(node)
        for attribute_type in unique_attribute_types:
            if node in attributes_dict:
                if attribute_type in attributes_dict[node]:
                    df_dict[attribute_type].append(attributes_dict[node][attribute_type])
                else:
                    df_dict[attribute_type].append("Unknown")
            else:
                print(node)
                df_dict[attribute_type].append("Unknown")
    
    return pd.DataFrame.from_dict(df_dict)

def open_file_browser(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", path])

def prune_subtree(node, allowed_labels, root_value="*"):
    """
    Prune the given 'node' so that:
      - If node's label is in allowed_labels (or node is root),
        we keep node (plus its pruned children).
      - Otherwise, bubble up any pruned children to the node's parent.
      - If no child is valid, the node is removed (None).

    Returns:
        - A single node dict if we keep the node,
        - A list of node dicts if we remove the node but keep its children,
        - None if everything under this node is removed.
    """
    # 1) Recursively prune children first (post-order)
    new_children = []
    for child in node["children"]:
        result = prune_subtree(child, allowed_labels, root_value)
        if result is None:
            # child and its subtree are all invalid => skip
            continue
        if isinstance(result, list):
            # child was removed but has valid children => bubble them up
            new_children.extend(result)
        else:
            # child is a single valid pruned node
            new_children.append(result)

    # 2) Decide if we keep 'node'
    is_root = (node["value"] == root_value)  # e.g., "*"
    is_allowed_label = (node.get("label") in allowed_labels)

    # Always keep the root, OR keep if label is allowed
    if is_root or is_allowed_label:
        # We keep this node, with pruned children
        pruned_node = {
            **node,  # copy existing keys if you want
            "children": new_children
        }
        return pruned_node
    else:
        # Node label not allowed => bubble its children up
        if new_children:
            return new_children  # Return a list so parent can attach them
        return None  # No valid children => entire subtree removed

def prune_tree(root_node, allowed_labels, root_value="*"):
    """
    Prune the entire tree starting from 'root_node'.
    
    If root is removed (not allowed AND no valid children), 
    we might end up with None or a list of top-level nodes. 
    Usually, if root_value == '*', we keep the root no matter what.
    
    Returns: 
       - A single pruned root node (typical case),
       - A list of nodes (if the root was removed or bubbled up children),
       - Or None if everything is pruned.
    """
    result = prune_subtree(root_node, allowed_labels, root_value)
    if result is None:
        # Entire tree pruned away
        return None
    if isinstance(result, list):
        # We ended up with a "forest" (multiple siblings at top level)
        return result
    # A single pruned root node
    return result

def get_attributes_from_df(df, column, attribute_dict):
    nodes = df[column]
    attributes_extracted = {}
    unique_attribute_types = []

    for node_name, attributes in attribute_dict.items():
        for attribute_type, value in attributes.items():
            if attribute_type not in attributes_extracted:
                attributes_extracted[attribute_type] = []
                unique_attribute_types.append(attribute_type)
    
    for node in nodes:
        for attribute_type in unique_attribute_types:
            if node not in attribute_dict:
                attributes_extracted[attribute_type].append("Unknown")
            else:
                if attribute_type not in attribute_dict[node]:
                    attributes_extracted[attribute_type].append("Unknown")
                else:
                    attributes_extracted[attribute_type].append(attribute_dict[node][attribute_type])
    
    return attributes_extracted


def predict_links_pykeen(model, 
                         training_triples_factory, 
                         prediction_direction, 
                         top_n_prediction, 
                         target_node, 
                         relation):

    if prediction_direction == "Tail":
        res = predict.predict_target(
        model=model,
        head=target_node, 
        relation=relation,
        triples_factory=training_triples_factory).filter_triples(training_triples_factory).df.head(top_n_prediction)

    elif prediction_direction == "Head":
        res = predict.predict_target(
        model=model,
        tail=target_node, 
        relation=relation,
        triples_factory=training_triples_factory).filter_triples(training_triples_factory).df.head(top_n_prediction)
    
    return res

def predict_links_telf(
        Xtilda,
        MASK,
        rows,
        cols,
        prediction_direction,
        top_n_prediction,
        target_node
):
    """
    Returns top predictions in DataFrame format.

    Parameters:
    Xtilda (np.array): Score matrix of shape (len(rows), len(cols))
    MASK (np.array): Mask matrix of shape (len(rows), len(cols)) with 0s for unknowns and 1s for knowns.
    rows (list of str): Row labels.
    cols (list of str): Column labels.
    prediction_direction (str): Either "head" or "tail", determines whether target_node is in rows or cols.
    top_n_prediction (int): Number of top predictions to return.
    target_node (str): Node to make predictions for.

    Returns:
    pd.DataFrame: DataFrame with columns ['head_label', 'relation', 'tail_label', 'score'].
    """

    # Identify the index based on prediction direction
    if prediction_direction == "Head":
        if target_node not in cols:
            raise ValueError(f"Target node {target_node} not found in columns.")
        target_index = cols.index(target_node)
        scores = Xtilda[:, target_index]  # Get scores for this column
        mask = MASK[:, target_index]  # Get mask for this column
        head_labels = rows  # Rows are head labels
        relation_label = "exist"
        tail_labels = [target_node] * len(rows)  # Tail is fixed

    elif prediction_direction == "Tail":
        if target_node not in rows:
            raise ValueError(f"Target node {target_node} not found in rows.")
        target_index = rows.index(target_node)
        scores = Xtilda[target_index, :]  # Get scores for this row
        mask = MASK[target_index, :]  # Get mask for this row
        head_labels = [target_node] * len(cols)  # Head is fixed
        relation_label = "exist"
        tail_labels = cols  # Columns are tail labels

    else:
        raise ValueError("prediction_direction must be either 'head' or 'tail'.")

    # Filter to unknowns using the mask
    unknown_indices = np.where(mask == 0)[0]
    filtered_scores = scores[unknown_indices]
    filtered_head_labels = [head_labels[i] for i in unknown_indices]
    filtered_tail_labels = [tail_labels[i] for i in unknown_indices]

    # Get top predictions
    top_indices = np.argsort(filtered_scores)[::-1][:top_n_prediction]  # Sort descending
    top_head_labels = [filtered_head_labels[i] for i in top_indices]
    top_tail_labels = [filtered_tail_labels[i] for i in top_indices]
    top_scores = [filtered_scores[i] for i in top_indices]

    # Construct result DataFrame
    df = pd.DataFrame({
        "head_label": top_head_labels,
        "relation": [relation_label] * len(top_scores),
        "tail_label": top_tail_labels,
        "score": top_scores
    })

    return df