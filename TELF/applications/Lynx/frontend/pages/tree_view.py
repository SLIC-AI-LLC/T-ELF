import streamlit as st
import streamlit.components.v1 as components
from pages.helpers.html_contents import render_tree
from pages.helpers.utils import (prune_tree, filter_documents,
                                 extract_unique_values)

st.set_page_config(layout="wide")
if ("project_loaded" in st.session_state and st.session_state.project_loaded == False) or "data" not in st.session_state:
    st.warning("⚠️ Data has not been loaded yet. Please load the data before accessing this page.")
    st.stop()

if st.session_state.view_type != "Document Analysis":
    st.warning("⚠️ Document Analysis view mode is not selected.")
    st.stop()

st.header(st.session_state.selected_project)
with st.sidebar:
    st.header("Token Search")

    top_n = st.selectbox("Top n:", ["All"] + list(range(1, 101)), index=0)
    keywords_search_type = st.selectbox("Word Search Type:", ["Keywords", "Denovo"], index=0)
    if keywords_search_type == "Keywords":
        token_search_field = "all_words"
    else:
        token_search_field = "all_words_denovo"
    
    # Gather all unique words
    words = list(
        set(
            w
            for wlist in st.session_state.documents_data[token_search_field].values()
            for w in wlist
        )
    )
    words.sort()

    selected_words = st.multiselect("Select words to search:", words)
    negative_words = st.multiselect("Select negative words:", words)
    token_search_type = st.radio("Search type:", ["and", "or"])

    # ---------------------------------------------------
    # Attribute Search
    st.header("Attribute Search")

    search_categories = {
        "Countries": {
            "options": extract_unique_values("all_countries", st.session_state.documents_data),
            "key": "country"
        },
        "Authors": {
            "options": extract_unique_values("all_authors", st.session_state.documents_data),
            "key": "author"
        },
        "Affiliations": {
            "options": extract_unique_values("all_affiliations", st.session_state.documents_data),
            "key": "affiliation"
        },
        f"Attribute {st.session_state.attribute_column}": {
            "options": extract_unique_values("all_attributes", st.session_state.documents_data),
            "key": "attributes"
        }
    }

    # Build UI for attribute selections
    attr_selections = {}
    for label, info in search_categories.items():
        with st.expander(f"Search by {label}"):
            attr_key = info["key"]
            attr_selections[f"selected_{attr_key}"] = st.multiselect(
                f"Select {label.lower()}:", info["options"]
            )
            attr_selections[f"{attr_key}_search_type"] = st.radio(
                "Search type:", ["and", "or"], key=f"{attr_key}_radio"
            )
    # ---------------------------------------------------
    filter_df_setting = False

    # ---------------------------------------------------
    # Display Settings
    st.header("Display")

    selected_contents = st.multiselect(
        "**Shown Content:**",
        options=["Keyword", "Denovo", "Author", "Affiliation", "Country", "Attribute"],
        default=["Keyword", "Country"]
    )
    top_n_content_shown = st.selectbox("**Top n shown:**", 
                                       ["All"] + list(range(1, 101)), index=10)

    with st.expander(f"Coloring"):
        selected_visibility = st.radio(
            "**Font Color:**",
            ["Dark", "Light"],
            index=0,
            horizontal=True,
        )
        contents_coloring = st.radio(
            "**Contents Coloring:**",
            ["No Color", "tab20", 
            "turbo", "hsv",
            "viridis", "plasma",
            "cividis"
            ],
            index=1,
            horizontal=True,
        )

any_token_selected = bool(selected_words)
any_negative_token_selected = bool(negative_words)
any_attr_selected = any(
    attr_selections.get(f"selected_{cfg['key']}", [])
    for cfg in search_categories.values()
)

# If nothing is selected, reset the data
if not any_token_selected and not any_attr_selected and not negative_words:
    st.session_state.data = st.session_state.data_original.copy()
    st.session_state.selected_indices = None
else:
    # Filter documents once based on both token and attribute selections
    selected_labels, selected_indices = filter_documents(
        data_map=st.session_state.data_map,
        top_n=top_n,
        selected_words=selected_words,
        token_search_type=token_search_type,
        attr_selections=attr_selections,
        search_categories=search_categories,
        keywords_search_type=keywords_search_type,
        use_index_map=filter_df_setting,
        negative_words=negative_words,
    )
    if selected_labels:
        st.session_state.data = [prune_tree(st.session_state.data_original[0], selected_labels, st.session_state.root_name)]
        st.session_state.selected_indices = selected_indices
    else:
        # No matches => clear data or handle how you prefer
        st.session_state.data = [{"children":[]}]
        st.session_state.selected_indices = None


#st.write(st.session_state.data_map["0"])
#st.write(st.session_state.data[0]["children"])
# display the tree
components.html(
    render_tree(tree_data=st.session_state.data[0]["children"], 
                data_map=st.session_state.data_map,
                settings = {
                    "selected_visibility":selected_visibility,
                    "selected_contents":selected_contents,
                    "top_n_content_shown":top_n_content_shown,
                    "contents_coloring":contents_coloring,
                }), 
    height=800, scrolling=True)