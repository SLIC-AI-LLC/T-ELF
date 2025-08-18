import streamlit as st
import os
import sys;sys.path.append(os.path.join("pages"))
import sys;sys.path.append(os.path.join("..", "backend"))
import sys;sys.path.append(os.path.join(".."))

if "project_loaded" not in st.session_state:
    st.session_state.project_loaded = False

load_project_page = st.Page(os.path.join("pages", "load_project.py"), title="Load Project", icon=":material/flag:", default=True)
tree_view_page = st.Page(os.path.join("pages", "tree_view.py"), title="Tree Search", icon=":material/allergy:", default=False)
document_analysis_view_page = st.Page(os.path.join("pages", "doc_view.py"), title="Document Analysis", icon=":material/lan:", default=False)
link_view_page = st.Page(os.path.join("pages", "link_view.py"), title="Link Prediction", icon=":material/linked_services:", default=False)

pg = st.navigation(
    {
        f"Lynx":[load_project_page],
        "Views":[tree_view_page, document_analysis_view_page, link_view_page],
    }
)
pg.run()
