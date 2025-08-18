import streamlit as st
import os

# Get file path from query parameters
if "html_figure" not in st.session_state:    
    st.warning("⚠️ No HTML figure has been loaded.")
    st.stop()  # This prevents the rest of the page from rendering

file_path = st.session_state.html_figure

if not file_path or not os.path.exists(file_path):
    st.error("Invalid or missing HTML file.")
else:
    #st.title(file_path)
    st.markdown(
        """
        <style>
            /* Remove padding/margins from Streamlit's default container */
            .block-container {
                padding: 100px 20px !important; /* Add space on all sides */
                margin: 0 auto !important;
                max-width: 100% !important;
            }

            /* Ensure iframe expands dynamically */
            iframe {
                display: block;
                width: calc(100% - 40px) !important; /* Adjust width to account for left/right padding */
                min-height: calc(100vh - 40px) !important; /* Adjust height to account for top/bottom padding */
                margin: 0 auto; /* Center iframe */
                border: none;
            }

            /* Allow the body to adjust to content size */
            html, body, .main {
                height: auto;
                min-height: 100vh;
                padding: 100px; /* Apply padding on all sides */
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Read and display HTML content
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(
        html_content,
        height=0,  # Handled by CSS
        scrolling=True  # Prevent unnecessary scrolling
    )