import streamlit as st
import os
import pandas as pd

from pages.helpers.load_project_data import (load_document_analysis_data, load_link_data, 
                                             load_config, set_configs)
from pages.helpers.utils import find_by_suffix, find_folder_by_prefix
from pages.helpers.displays import (display_view_type_setting, display_file_node_settings,
                                    display_attribute_text_settings)

# Set the path of the Projects folder
PROJECTS_FOLDER = "projects"  # Change this to your actual projects folder path

# Function to get all project folders
def get_project_folders(root_path):
    if not os.path.exists(root_path):
        return []
    return sorted([folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))])

# Get list of projects
project_folders = get_project_folders(PROJECTS_FOLDER)

st.title("📂 Select a Project")

st.divider()

if not project_folders:
    st.warning("No projects found in the Projects folder.")
else:

    # Use selectbox for a modern dropdown (scrollable & searchable)
    selected_project = st.selectbox("Projects:", project_folders)


    # Get full path of the selected project
    selected_project_path = os.path.join(PROJECTS_FOLDER, selected_project)

    #
    # SETTINGS
    #
    if selected_project_path:
        st.divider()
        st.subheader("Settings")
        
        settings_file = find_by_suffix(os.path.join(selected_project_path), suffix="settings", ends=False)
        if settings_file:
            available_settings_setup = ["Manual", "Automatic"]
            settings_setup_index = 1
        else:
            available_settings_setup = ["Manual"]
            settings_setup_index = 0

        selected_settings_setup = st.radio(
            "**Settings Setup:**",
            available_settings_setup,
            index=settings_setup_index,
            horizontal=True,
        )

        if "configuration" in st.session_state:
            configuration = st.session_state.configuration
            if configuration["selected_project_path"] != selected_project_path:
                configuration = None
                del st.session_state.configuration
        else:
            configuration = None

        if selected_settings_setup == "Automatic":
            configuration = load_config(settings_file)
            configuration["selected_project_path"] = selected_project_path
            st.session_state.configuration = configuration
            set_configs(configuration)
            with st.expander(f"Loaded Settings:"):
                st.table(configuration)

        if selected_settings_setup == "Manual":
            display_view_type_setting(configuration=configuration)
            if "view_type" in st.session_state and st.session_state.view_type == "Document Analysis":
                display_file_node_settings(configuration=configuration)
            
        #
        # DO LOADING OF THE PROJECT
        #
        if "view_type" in st.session_state and st.session_state.view_type == "Document Analysis":
            
            try:
                st.session_state.folder_name = find_folder_by_prefix(os.path.join(selected_project_path, "depth_0", st.session_state.root_name), prefix="0")
            except:
                st.warning(f"⚠️ Can't find the directory at {os.path.join(selected_project_path, 'depth_0', st.session_state.root_name)}")
                st.stop()

            file = find_by_suffix(os.path.join(selected_project_path, "depth_0", st.session_state.root_name, st.session_state.folder_name), suffix=st.session_state.document_file_name_suffix, ends=False)
            
            if file is None:
                st.warning(f"⚠️ Document files with suffix **```{st.session_state.document_file_name_suffix}```** does not exist at path **```{os.path.join(selected_project_path, 'depth_0', st.session_state.root_name, folder_name)}```**!")
                st.stop()

            st.session_state.available_columns = pd.read_csv(file).columns.to_list()
            if selected_settings_setup == "Manual":
                display_attribute_text_settings(configuration=configuration)
            else:
                if st.session_state.text_column not in st.session_state.available_columns:
                    st.warning(f"The text column {st.session_state.text_column} not in the available columns!")

                if st.session_state.attribute_column not in st.session_state.available_columns:
                    st.warning(f"The attribute column {st.session_state.attribute_column} not in the available columns!")

            with st.expander(f"Available Columns in `{file}`:"):
                st.write(st.session_state.available_columns)

                if st.session_state.text_column not in st.session_state.available_columns:
                    st.warning(f"⚠️ Text column {st.session_state.text_column} does not exist!")
                if st.session_state.attribute_column not in st.session_state.available_columns:
                    st.warning(f"⚠️ Attribute column {st.session_state.attribute_column} does not exist!")
                

            with st.spinner(text="In progress...", show_time=False):

                    st.session_state.data, st.session_state.data_map, st.session_state.documents_data = load_document_analysis_data(
                        selected_project_path, 
                        st.session_state.text_column,
                        st.session_state.document_file_name_suffix,
                        st.session_state.attribute_column,
                        st.session_state.attribute_column_split,
                        st.session_state.root_name
                    )
                    st.session_state.data_original = st.session_state.data.copy()
                    st.session_state.selected_indices = None
                    st.session_state.project_loaded = True
        
        elif "view_type" in st.session_state and st.session_state.view_type == "Link Prediction":
            st.session_state.data = load_link_data(selected_project_path)

            if st.session_state.data is None and not isinstance(st.session_state.data, dict):
                st.warning("⚠️ Not a known Link Prediction project type!")
                st.session_state.project_loaded=False
                st.stop()

            st.session_state.project_loaded = True
    
        else:
            st.session_state.project_loaded = False
            st.warning(f"View type is not implemented yet!") 

        #
        # AFTER LOADED
        #
        if st.session_state.project_loaded:
            st.session_state.selected_project = selected_project
            st.session_state.selected_project_path = selected_project_path
            st.success(f"**Project:** `{st.session_state.selected_project}` at `{st.session_state.selected_project_path}`")
