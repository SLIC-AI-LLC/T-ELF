import streamlit as st
from pages.helpers.utils import (extract_attribute_lists, 
                                 extract_link_attributes_df,
                                 predict_links_pykeen,
                                 get_attributes_from_df,
                                 predict_links_telf)
from pages.helpers.displays import (display_attributes,
                                    display_link_node_search)
import pandas as pd 
st.set_page_config(layout="wide")
if ("project_loaded" in st.session_state and st.session_state.project_loaded == False) or "data" not in st.session_state:
    st.warning("⚠️ Data has not been loaded yet. Please load the data before accessing this page.")
    st.stop()

if st.session_state.view_type != "Link Prediction":
    st.warning("⚠️ Link Prediction mode is not selected.")
    st.stop()

st.header(st.session_state.selected_project)
st.markdown(f"**{st.session_state.data['type'].upper()} project**")

#
### ATTRIBUTES
#
st.subheader("Attributes Display")
attribute_lists = extract_attribute_lists(st.session_state.data["data"]["attributes"])

with st.expander("All Attribute Stats"):
    attribute_lists_tabs = st.tabs(attribute_lists.keys())
    for idx, (attribute_name, all_attributes) in enumerate(attribute_lists.items()):
        with attribute_lists_tabs[idx]:
            display_attributes(all_attributes, key=f"link_attribute_{attribute_name}")

st.divider()

#
### NODES
#
st.subheader("Node Search")
link_attributes_df = extract_link_attributes_df(
        st.session_state.data["data"]["rows"],
        st.session_state.data["data"]["cols"],
        st.session_state.data["data"]["attributes"]
)
display_link_node_search(link_attributes_df)
st.divider()

#
### PREDICTIONS
#
st.subheader("Predict Link")
prediction_settings_columns = st.columns(3)
with prediction_settings_columns[0]:
    prediction_direction = st.selectbox("Prediction Type", options=["Tail", "Head"])
with prediction_settings_columns[1]:
    top_n_prediction = st.number_input("Top n Predictions", min_value=1, value=10)
with prediction_settings_columns[2]:
    relation = st.selectbox("Relation", 
                            options=st.session_state.data["data"]["triples"]["relation"].unique(), 
                            disabled=st.session_state.data['type'] != "pykeen")

if prediction_direction == "Tail":
    prediction_options = st.session_state.data["data"]["rows"]
elif prediction_direction == "Head":
    prediction_options = st.session_state.data["data"]["cols"]
target_node = st.selectbox(f"Select {prediction_direction} Node", options=prediction_options)

if st.session_state.data['type'] == "pykeen":
    predictions_df = predict_links_pykeen(
        model=st.session_state.data["model"], 
        training_triples_factory=st.session_state.data["training_triples_factory"], 
        prediction_direction=prediction_direction, 
        top_n_prediction=top_n_prediction, 
        target_node=target_node, 
        relation=relation
    )

elif st.session_state.data['type'] == "telf":
    predictions_df = predict_links_telf(
        Xtilda=st.session_state.data["data"]["Xtilda"],
        MASK=st.session_state.data["data"]["MASK"],
        rows=st.session_state.data["data"]["rows"],
        cols=st.session_state.data["data"]["cols"],
        prediction_direction=prediction_direction,
        top_n_prediction=top_n_prediction, 
        target_node=target_node, 
    )

st.markdown("**Predictions:**")
st.dataframe(predictions_df, use_container_width=True)
prediction_attributes = get_attributes_from_df(
    predictions_df, 
    column=f"{prediction_direction.lower()}_label", 
    attribute_dict=st.session_state.data["data"]["attributes"])

with st.expander("Prediction Attribute Stats"):
    prediction_attribute_tabs = st.tabs(prediction_attributes.keys())
    for idx, (attribute_name, all_attributes) in enumerate(prediction_attributes.items()):
        with prediction_attribute_tabs[idx]:
            display_attributes(all_attributes, key=f"link_prediction_attribute_{attribute_name}")

st.markdown("**Known relations:**")
if prediction_direction == "Head":
    filter_known_df = st.session_state.data["data"]["triples"][st.session_state.data["data"]["triples"]["tail"] == target_node]
    st.dataframe(
        filter_known_df,
        use_container_width=True
    )
elif prediction_direction == "Tail":
    filter_known_df = st.session_state.data["data"]["triples"][st.session_state.data["data"]["triples"]["head"] == target_node]
    st.dataframe(
        filter_known_df,
        use_container_width=True
    )

prediction_known_attributes = get_attributes_from_df(
    filter_known_df, 
    column=f"{prediction_direction.lower()}", 
    attribute_dict=st.session_state.data["data"]["attributes"])

with st.expander("Known Attribute Stats"):
    prediction_known_attribute_tabs = st.tabs(prediction_known_attributes.keys())
    for idx, (attribute_name, all_attributes) in enumerate(prediction_known_attributes.items()):
        with prediction_known_attribute_tabs[idx]:
            display_attributes(all_attributes, key=f"link_prediction_known_attribute_{attribute_name}")

        