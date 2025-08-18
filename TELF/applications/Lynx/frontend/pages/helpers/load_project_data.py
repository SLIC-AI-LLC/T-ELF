import streamlit as st
import os
import re
from pages.helpers.utils import (split_and_flatten_list, load_csv_file_items,
                                 extract_affiliation_info, get_top_words,
                                 find_folder_by_prefix, file_exists_in_path,
                                 get_token_index_map)
from pykeen.triples import TriplesFactory
import torch
import pickle
import json

@st.cache_resource
def load_link_data(path):
    
    # PYKEEN TYPE
    if find_folder_by_prefix(path, "pykeen_model") and file_exists_in_path(path, "data.p"):
        # Under the project must have pykeen_model folder (results.save_to_directory()), and data.p
        # data.p is a dictinary of {rows:list, columns:list, attributes:dict, triples:df}
        # attributes:dict keys are the node names from rows and cols, and value of the attributes:dict are dictionary where key is attribute type and value is the attribute value
        # Example attributes:dict = {"Node 1":{"country":"USA"}}
        # triples:df is a dataframe with columns head, relation, and tail where head is from rows, and tail is from columns.
        model = torch.load(os.path.join(path, "pykeen_model", "trained_model.pkl"), weights_only=False, map_location=torch.device('cpu'))
        training_triples_factory = TriplesFactory.from_path_binary((os.path.join(path, "pykeen_model", "training_triples")))
        data = pickle.load(open(os.path.join(path, "data.p"), "rb"))
        return {"data":data, "training_triples_factory":training_triples_factory, "model":model, "type":"pykeen"}
    
    # TELF TYPE
    elif file_exists_in_path(path, "telf.p"):
        # Under the project must have telf.p
        # telf.p is a dictionary of {rows:list, columns:list, attributes:dict, triples:df, Xtilda:np.ndarray, MASK:np.ndarray}
        # attributes:dict keys are the node names from rows and cols, and value of the attributes:dict are dictionary where key is attribute type and value is the attribute value
        # Example attributes:dict = {"Node 1":{"country":"USA"}}
        # triples:df is a dataframe with columns head, relation, and tail where head is from rows, and tail is from columns.
        data = pickle.load(open(os.path.join(path, "telf.p"), "rb"))
        return {"data":data, "type":"telf"}
    
    # NOT A Link Prediction PROJECT
    else:
        # todo
        return None

@st.cache_data
def load_document_analysis_data(path, text_column, document_file_name_suffix, attribute_column, attribute_column_split, root_name):
    graph = build_graph_from_directory(path, root_name)
    tree = build_tree(graph, root_name)
    documents_data, tree_map = extract_data_from_graph(graph, path, 
                                                       text_column, 
                                                       document_file_name_suffix,
                                                       attribute_column, attribute_column_split)
    
    return tree, tree_map, documents_data

def extract_data_from_graph(graph, path, text_column, document_file_name_suffix, attribute_column, attribute_column_split):
    """Extracts unigrams, authors, affiliations, and countries from the graph data."""
    documents_data = {
        "all_words": {},
        "all_words_denovo":{},
        "all_authors": {},
        "all_affiliations": {},
        "all_countries": {},
        "all_text":{},
        "all_attributes":{}
    }
    tree_map = {}

    for node in graph:
        node_label = f"{node['name']}-{node['label']} ({node['document_count']})"

        # Extract unigrams
        try:
            unigrams = load_csv_file_items(
                path=node["path"],
                suffix="_bow_unigrams.csv",
                ends=True,
                column="word"
            )
        except:
            unigrams = []

        if unigrams is not None:
            documents_data["all_words"][node_label] = list(unigrams)

        # Extract authors
        try:
            authors_raw, authors_index_map = split_and_flatten_list(load_csv_file_items(
                path=node["path"],
                suffix=document_file_name_suffix,
                ends=False,
                column="author_ids"
            ), use_index_map=True)
            authors = list(set(authors_raw))
        except:
            authors = []

        if authors is not None:
            documents_data["all_authors"][node_label] = authors

        # Extract affiliations and countries
        try:
            affiliations = load_csv_file_items(
                path=node["path"],
                suffix=document_file_name_suffix,
                ends=False,
                column="affiliations"
            )
            organizations, countries, org_index_map, country_index_map = extract_affiliation_info(affiliations, use_index_map=True)
        except:
            organizations = []
            countries = []

        documents_data["all_affiliations"][node_label] = organizations
        documents_data["all_countries"][node_label] = countries

        # Extract denovo words
        try:
            all_text = load_csv_file_items(
                path=node["path"],
                suffix=document_file_name_suffix,
                ends=False,
                column=str(text_column)
            )
        except:
            all_text = []

        documents_data["all_text"][node_label] = all_text
        denovo_unigrams = get_top_words(all_text, top_n=50, n_gram=1, verbose=False, filename=None)["word"]
        denovo_unigrams_index_map = get_token_index_map(all_text, list(denovo_unigrams))
        documents_data["all_words_denovo"][node_label] = list(denovo_unigrams)

        # Extract attributes
        try:
            all_attributes, attributes_index_map = split_and_flatten_list(load_csv_file_items(
                path=node["path"],
                suffix=document_file_name_suffix,
                ends=False,
                column=str(attribute_column)
            ), split_by=attribute_column_split, use_index_map=True)
        except:
            all_attributes = []
        documents_data["all_attributes"][node_label] = all_attributes

        # Store data in tree_map
        tree_map[node["name"]] = {
            "path": node["path"],
            "label": node_label,
            "unigrams": unigrams,
            "denovo_unigrams":denovo_unigrams,
            "denovo_unigrams_index_map":denovo_unigrams_index_map,
            "author": authors,
            "author_index_map": authors_index_map,
            "affiliation": organizations,
            "affiliation_index_map": org_index_map,
            "country": countries,
            "country_index_map": country_index_map,
            "all_text": all_text,
            "attributes":all_attributes,
            "attributes_index_map":attributes_index_map,
        }

    return documents_data, tree_map


def build_tree(nodes, root_name):
    node_dict = {node["name"]: node for node in nodes}
    tree = []

    def add_children(node_name):
        node = node_dict[node_name]
        label = f"{node['name']}-{node['label']} ({node['document_count']})"
            
        return {
            "label": label,
            "value": node_name,
            "children": [add_children(child) for child in node["children"]]
        }

    # build from root node(s)
    for node in nodes:
        if node["name"] == root_name:  # or any other condition for your root
            tree.append(add_children(node["name"]))
    return tree

def parse_topic_folder_name(folder_name, path=None):
    """
    Extracts topic number, label, and document count from folder name.

    Expected format: topic_number-label_with_underscores-documents_count-documents
    Example: '3-label_of_the_topic_9-documents' -> ('3', 'Label of the topic', '9')

    Args:
        folder_name (str): Folder name formatted as: <topic_number>-<label_with_underscores>_<document_count>-documents

    Returns:
        tuple: (str, str, str) -> (topic_number, cleaned_label, document_count)
               or (None, None, None) if parsing fails.
    """
    match = re.match(r'(\d+)-([^-]+)_(\d+)-documents', folder_name)
    if match:
        topic_number = match.group(1)  # Extract topic number
        label = match.group(2).replace("_", " ").strip()  # Replace underscores with spaces
        document_count = match.group(3)  # Extract document count
        return topic_number, label, document_count
    elif folder_name.isdigit():
        labels = load_csv_file_items(path, suffix="cluster_summaries", ends=False, column="label")
        if path is None or labels is None:
            return folder_name, "Topic", "Unknown"
        else:
            return folder_name, labels[int(folder_name)], "Unknown"

    return None, None, None


def map_folder_to_logical_name(folder_name: str, root_name) -> str:
    """
    Convert a physical folder name (like '*_0', '*_1_2', or '0_1')
    to a 'logical' node name.

    Examples:
        map_folder_to_logical_name("*")       -> "*"
        map_folder_to_logical_name("*_0")     -> "0"
        map_folder_to_logical_name("*_1_5")   -> "1_5"
        map_folder_to_logical_name("1_2")     -> "1_2"
        map_folder_to_logical_name("0")       -> "0"

    The idea is to strip a leading '*_' if present, because
    '*_0' means "the node for root topic 0."
    """
    if folder_name == root_name:
        return root_name
    # If it starts with "*_", remove it
    if folder_name.startswith(f"{root_name}_"):
        return folder_name[len(root_name)+1:]  # remove the '*_' prefix
    return folder_name


def get_logical_parent_name(logical_name: str, root_name):
    """
    Given a logical name (like '0', '1_2', '1_2_3'), return its parent.

    Rules:
    - If the logical name is "*", it's the root (no parent).
    - If there's only one chunk and it's not "*", parent is "*".
    - Otherwise, drop the last chunk.

    Examples:
        get_logical_parent_name("*")      -> None
        get_logical_parent_name("0")      -> "*"
        get_logical_parent_name("1_5")    -> "1"
        get_logical_parent_name("1_5_0")  -> "1_5"
    """

    if logical_name == root_name:
        return None  # root

    parts = logical_name.split("_")
    if len(parts) == 1:
        # single chunk (e.g. "0", "1") => parent is "*"
        return root_name
    else:
        # multiple chunks => drop the last chunk
        return "_".join(parts[:-1])


def build_graph_from_directory(root_path, root_name):
    """
    Parses the hierarchical directory structure to build a list of nodes.

    Each folder in `depth_i` is considered a "container folder".
    Inside that folder, we expect subfolders that match the topic pattern:
        <topic_number>-<label_with_underscores>_<document_count>-documents

    We map each container folder to a 'logical' name, and
    each topic subfolder becomes a child node of that container folder.

    Args:
        root_path (str): Path to the root directory.

    Returns:
        list: A list of dictionaries representing the nodes in the graph.
    """
    # Dictionary of node_name -> node_data
    nodes = {}

    # Ensure we have a root node named "*", or root_name
    if root_name not in nodes:
        nodes[root_name] = {
            "name": root_name,
            "label": "Root",
            "parent": None,
            "children": [],
            "document_count": None,
            "path": root_path
        }

    # Iterate through depth_* folders
    for depth_folder in sorted(os.listdir(root_path)):
        depth_path = os.path.join(root_path, depth_folder)
        if not os.path.isdir(depth_path) or not depth_folder.startswith("depth_"):
            continue  # ignore non-depth folders
        # Iterate through all container folders at this depth
        for physical_folder_name in sorted(os.listdir(depth_path)):
            container_folder_path = os.path.join(depth_path, physical_folder_name)
            if not os.path.isdir(container_folder_path):
                continue  # skip files
            
            # Map physical folder to a logical node name
            container_logical_name = map_folder_to_logical_name(physical_folder_name, root_name)

            # Ensure the container folder node itself exists
            if container_logical_name not in nodes:
                # figure out its parent
                parent_name = get_logical_parent_name(str(container_logical_name), root_name)
                
                nodes[container_logical_name] = {
                    "name": container_logical_name,
                    "label": None,  # might not be a "topic" node but a container
                    "parent": parent_name,
                    "children": [],
                    "document_count": None,
                    "path": container_folder_path
                }
                # link to parent
                if parent_name and parent_name not in nodes:
                    # create placeholder for parent's node if missing
                    nodes[parent_name] = {
                        "name": parent_name,
                        "label": None,
                        "parent": get_logical_parent_name(str(parent_name), root_name),
                        "children": [],
                        "document_count": None,
                        "path": None
                    }
                if parent_name:
                    nodes[parent_name]["children"].append(container_logical_name)

            # Now check all "topic subfolders" in this container
            topic_folders = sorted(
                f for f in os.listdir(container_folder_path)
                if os.path.isdir(os.path.join(container_folder_path, f))
            )
            
            for topic_folder in topic_folders:
                topic_number, label, document_count = parse_topic_folder_name(topic_folder, container_folder_path)
                if topic_number is None:
                    # Not a recognized topic folder; skip
                    continue

                # Build the full node name for the topic
                if container_logical_name == root_name:
                    # If container is the global root "*"
                    # then the node name is just the topic_number
                    topic_node_name = topic_number
                else:
                    # Otherwise, we append the topic_number to the container's logical name
                    topic_node_name = f"{container_logical_name}_{topic_number}"

                topic_path = os.path.join(container_folder_path, topic_folder)

                # If this topic node doesn't exist yet, create it
                if topic_node_name not in nodes:
                    nodes[topic_node_name] = {
                        "name": topic_node_name,
                        "label": label,
                        "parent": container_logical_name,  # parent is the container folder node
                        "children": [],
                        "document_count": document_count,
                        "path": topic_path
                    }

                    # Add it as a child of the container folder node
                    if container_logical_name not in nodes:
                        # container might not have been created yet if it's missing
                        # but we created it above, so this is just a safety check
                        nodes[container_logical_name] = {
                            "name": container_logical_name,
                            "label": None,
                            "parent": get_logical_parent_name(str(container_logical_name), root_name),
                            "children": [],
                            "document_count": None,
                            "path": container_folder_path
                        }
                    nodes[container_logical_name]["children"].append(topic_node_name)

    # Convert node dictionary to a list
    return list(nodes.values())

def load_config(file_path):
  """
  Loads a JSON configuration file.

  Args:
    file_path: The path to the JSON file.

  Returns:
    A dictionary containing the configuration data, or None if the file
    could not be loaded.
  """
  try:
    with open(file_path, 'r') as f:
      config = json.load(f)
    return config
  except FileNotFoundError:
    st.warning(f"Error: File not found at {file_path}")
    return None
  except json.JSONDecodeError:
    st.warning(f"Error: Invalid JSON format in {file_path}")
    return None

def set_configs(configuration):
    """
    Sets Streamlit session state variables from a configuration dictionary,
    checking for key existence first.
    """
    if "VIEW_TYPE" in configuration:
        st.session_state.view_type = configuration["VIEW_TYPE"]
    if "DOCUMENT_FILE_START_WITH" in configuration:
        st.session_state.document_file_name_suffix = configuration["DOCUMENT_FILE_START_WITH"]
    if "ROOT_NAME" in configuration:
        st.session_state.root_name = configuration["ROOT_NAME"]
    if "TEXT_COLUMN" in configuration:
        st.session_state.text_column = configuration["TEXT_COLUMN"]
    if "ATTRIBUTE_COLUMN" in configuration:
        st.session_state.attribute_column = configuration["ATTRIBUTE_COLUMN"]
    if "ATTRIBUTE_COLUMN_SPLIT_BY" in configuration:
        st.session_state.attribute_column_split = configuration["ATTRIBUTE_COLUMN_SPLIT_BY"]