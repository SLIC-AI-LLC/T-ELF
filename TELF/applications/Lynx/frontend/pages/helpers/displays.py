import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components
import os
from pages.helpers.utils import (find_by_suffix, open_file_browser, 
                                 get_top_words, preprocess_authors,
                                 build_coauthorship_network, extract_all_affiliation_info)
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
from collections import Counter
import networkx as nx

def st_normal():
    _, col, _ = st.columns([1, 2, 1])
    return col

def display_html(path, height=800):
    st.markdown(
        """
        <style>
            .full-width-container {
                position: fixed;
                left: 0;
                width: 100%;
                bottom: 0;
                background-color: white;
                z-index: 9999;
                box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
                padding: 10px;
            }
            .full-width-iframe {
                width: 100%;
                height: 800px; /* Reasonable default height */
                border: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
    st.markdown(f"**{os.path.basename(path)}**")
    components.html(html_content, height=height, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_denovo_words(data, key):
    st.subheader(f"De novo words from '{st.session_state.text_column}' column")
    n_gram=st.slider(
        "n_gram:",
        min_value=1,
        max_value=5,
        value=2,
        key=f"{key}_slider"
    )
    st.dataframe(get_top_words(data, 
                                top_n=50, 
                                n_gram=n_gram,
                                    verbose=False, filename=None)[["word", "tf", "df"]],
    use_container_width=True)

def display_coauthorship_heatmap(directory, suffix, ends=False, 
                                 columns=["authors", "author_ids"], 
                                 title="Co-authorship Heatmap", key=""):
    """
    Generates an interactive heatmap of co-authorship counts using Plotly with dynamic filtering.
    Filters Y-axis based on user selection and X-axis dynamically adjusts to show only relevant collaborators.
    """
    st.subheader(title)
    file = find_by_suffix(directory, suffix=suffix, ends=ends)
    
    if file:
        df = pd.read_csv(file)
        for col in columns:
            if col not in df.columns.tolist():
                st.warning(f"⚠️ {col} not in the DataFrame!")
                return
    else:
        st.warning(f"⚠️ CSV not found!")
        return
    
    coauthorship_counts = {}

    # Process co-authors and count occurrences
    for authors, ids in zip(df[columns[0]], df[columns[1]]):
        authors_list = preprocess_authors([authors], [ids])

        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                pair = tuple(sorted([authors_list[i], authors_list[j]]))
                coauthorship_counts[pair] = coauthorship_counts.get(pair, 0) + 1

    # Convert to DataFrame
    coauthorship_df = pd.DataFrame(list(coauthorship_counts.items()), columns=["Pair", "Count"])
    coauthorship_df[["Author 1", "Author 2"]] = coauthorship_df["Pair"].apply(pd.Series)

    # Pivot table for heatmap
    coauthorship_matrix = coauthorship_df.pivot(index="Author 1", columns="Author 2", values="Count").fillna(0)

    # Extract unique author names
    all_authors = sorted(set(coauthorship_df["Author 1"]).union(set(coauthorship_df["Author 2"])))

    # Create a searchable dropdown for author selection (Y-axis)
    selected_authors = st.multiselect(
        "🔍 Select Author(s) to Filter Y-Axis:",
        all_authors,
        default=[]  # No default selection
    )

    # Filter Y-axis based on selection
    if selected_authors:
        filtered_matrix = coauthorship_matrix.loc[selected_authors, :]
    else:
        filtered_matrix = coauthorship_matrix  # Show all if no selection

    # **Dynamic Filtering for X-axis: Keep only actual collaborators**
    if not filtered_matrix.empty:
        # Get all actual collaborators of selected authors
        relevant_x_authors = filtered_matrix.columns[filtered_matrix.sum(axis=0) > 0].tolist()
        filtered_matrix = filtered_matrix.loc[:, relevant_x_authors]  # Filter X-axis
    else:
        relevant_x_authors = coauthorship_matrix.columns.tolist()  # Show all if empty

    # Convert filtered matrix to long-form data for Plotly
    filtered_long = filtered_matrix.stack().reset_index()
    filtered_long.columns = ["Author 1", "Author 2", "Count"]

    # Plot interactive Heatmap using Plotly
    fig = px.imshow(
        filtered_matrix.values,
        labels=dict(x="Co-Author", y="Filtered Author(s)", color="Collaboration Count"),
        x=filtered_matrix.columns,
        y=filtered_matrix.index,
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        xaxis_title="Filtered Co-Authors",
        yaxis_title="Selected Authors",
        width=900, height=700
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_coauthor_heatmap")



def display_interactive_coauthorship_plot(directory, suffix, ends=False, 
                             columns=["authors", "author_ids"], 
                             title="Co-authorship Network", key=""):
    """
    Streamlit function to display an interactive co-authorship network plot.
    """
    st.subheader(title)
    file = find_by_suffix(directory, suffix=suffix, ends=ends)
    if file:
        df = pd.read_csv(file)
        for col in columns:
            if col not in df.columns.tolist():
                st.warning(f"⚠️ {col} not in the DataFrame!")
                return
    else:
        st.warning(f"⚠️ CSV not found!")
        return
    # Build graph
    G = build_coauthorship_network(df[columns[0]], df[columns[1]])

    # Dropdown for selecting an author
    author_names = list(G.nodes)
    selected_author = st.selectbox("Select an author to highlight:", [""] + author_names, key=f"{key}_selectbox1")

    # Dropdown for layout selection
    layout_options = {
        "Spring Layout": nx.spring_layout,
        "Kamada-Kawai Layout": nx.kamada_kawai_layout,
        "Circular Layout": nx.circular_layout,
        "Fruchterman-Reingold Layout": nx.fruchterman_reingold_layout,
        "Random Layout": nx.random_layout,
        "Shell Layout": nx.shell_layout
    }
    selected_layout = st.selectbox("Select network layout:", list(layout_options.keys()), index=0, key=f"{key}_selectbox2")

    # Compute node positions based on selected layout
    pos = layout_options[selected_layout](G)

    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]["weight"]

        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            line=dict(width=max(0.5, weight / 2), color="gray"),  
            hoverinfo="text",
            mode="lines",
            text=f"Co-authored {weight} times"
        )
        edge_traces.append(edge_trace)

    # Create node traces
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        if node == selected_author:
            node_color.append("red")  # Highlight selected author
        else:
            node_color.append("blue")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=10, color=node_color, line=dict(width=2, color="black")),
        hoverinfo="text"
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Co-authorship Network Graph",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_coauthor_network")


def display_author_chart(directory, suffix, ends=False, 
                             columns=["authors", "author_ids"], 
                             title="Author Distribution Chart", key=""):
    """
    Streamlit function to display an interactive pie chart of author distributions using Plotly.
    """
    st.subheader(title)
    file = find_by_suffix(directory, suffix=suffix, ends=ends)
    if file:
        df = pd.read_csv(file)
        for col in columns:
            if col not in df.columns.tolist():
                st.warning(f"⚠️ {col} not in the DataFrame!")
                return
            
        # Preprocess author names
        author_list = preprocess_authors(df[columns[0]], df[columns[1]])

        # Count occurrences
        author_counts = pd.Series(author_list).value_counts().reset_index()
        author_counts.columns = ["Author", "Count"]

        # Calculate percentage share
        total_publications = author_counts["Count"].sum()
        author_counts["Share (%)"] = (author_counts["Count"] / total_publications) * 100

        # Select number of top authors to display
        top_n = st.slider("Select number of top authors to display:", 1, 100, 1, key=f"{key}_slider")
        author_counts = author_counts.head(top_n)

        # Create interactive bar chart using Plotly
        fig = px.bar(
            author_counts,
            x="Author",
            y="Count",
            text="Share (%)",  # Show percentage on bars
            title="Top Authors by Number of Publications (with Share %)",
            labels={"Author": "Author Name", "Count": "Number of Publications"},
            hover_data={"Share (%)": ":.2f"},
        )

        # Adjust layout for better readability
        fig.update_layout(
            xaxis_tickangle=-45, 
            showlegend=False, 
            yaxis_title="Number of Publications",
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_author_chart")
    else:
        st.warning(f"⚠️ CSV not found!")

def display_document_affiliations(directory, suffix, title, ends, column="affiliations", warning_message="CSV not found!", key=""):
    file = find_by_suffix(directory, suffix=suffix, ends=ends)
    if file:
        st.subheader(title)
        df = pd.read_csv(file)
        if column not in df.columns.tolist():
            st.warning(f"⚠️ {column} not in the DataFrame!")
            return
    else:
        st.warning(f"⚠️ CSV not found!")
        return
    organizations, countries = extract_all_affiliation_info(df[column])
    # organizations
    org_counts = pd.Series(organizations).value_counts().reset_index()
    org_counts.columns = ["Organization", "Count"]
    top_n_orgs = st.slider("Select number of top organizations to display:", 1, min(100, len(org_counts)), 1, key=f"{key}_slider1")
    org_counts = org_counts.head(top_n_orgs)
    fig_orgs = px.bar(
        org_counts,
        x="Organization",
        y="Count",
        text="Count",
        title="Top Organizations by Count",
        labels={"Organization": "Organization Name", "Count": "Number of Publications"},
    )
    fig_orgs.update_layout(xaxis_tickangle=-45, showlegend=False)

    # Display both plots in Streamlit
    st.plotly_chart(fig_orgs, use_container_width=True, key=f"{key}_plotly_organization_count")

    # countries
    country_counts = pd.Series(countries).value_counts().reset_index()
    country_counts.columns = ["Country", "Count"]
    top_n_countries = st.slider("Select number of top countries to display:", 1, min(100, len(country_counts)), 1, key=f"{key}_slider2")
    country_counts = country_counts.head(top_n_countries)
    fig_countries = px.bar(
        country_counts,
        x="Country",
        y="Count",
        text="Count",
        title="Top Country by Count",
        labels={"Country": "Country Name", "Count": "Number of Publications"},
    )
    fig_orgs.update_layout(xaxis_tickangle=-45, showlegend=False)

    # Display both plots in Streamlit
    st.plotly_chart(fig_countries, use_container_width=True, key=f"{key}_plotly_country_count")

def display_document_years(directory, suffix, title, ends, 
                           column="year", warning_message="CSV not found!", key="",
                           index_filter=None):
    file = find_by_suffix(directory, suffix=suffix, ends=ends)
    if file:
        st.subheader(title)
        df = pd.read_csv(file)
        if column not in df.columns.tolist():
            st.warning(f"⚠️ {column} not in the DataFrame!")
            return
        
        # ------------------ filter rows by index ------------------
        if index_filter is not None:
            # Ensure we’re working with a set for fast lookup
            wanted = set(index_filter)
            df = df.loc[df.index.intersection(wanted)]
        
        # Step 1: Convert "Year" column to numeric, handling errors
        df[column] = pd.to_numeric(df[column], errors="coerce")  # Convert invalid years to NaN
        df = df.dropna(subset=[column])  # Remove rows with NaN values
        df[column] = df[column].astype(int)  # Ensure integer type

        # Step 2: Count occurrences of each year
        year_counts = df[column].value_counts().sort_index()

        # Step 3: Ensure no missing years by creating a complete range
        year_range = np.arange(year_counts.index.min(), year_counts.index.max() + 1)
        df_counts = pd.DataFrame({"Year": year_range, "Value": year_counts.reindex(year_range, fill_value=0)})

        if df_counts["Year"].min() == df_counts["Year"].max():
            filtered_df = df_counts
        else:
            year_range_selection = st.slider(
                "Select Year Range:", 
                min_value=df_counts["Year"].min(), 
                max_value=df_counts["Year"].max(), 
                value=(df_counts["Year"].min(), df_counts["Year"].max()),
                key=f"{key}_slider"
            )

            # 🏷️ Filter DataFrame based on selected years
            filtered_df = df_counts[(df_counts["Year"] >= year_range_selection[0]) & (df_counts["Year"] <= year_range_selection[1])]

        # 📊 Create an interactive Plotly line chart
        fig = px.line(
            filtered_df, 
            x="Year", 
            y="Value", 
            labels={"Year": "Year", "Value": "Count"},
            markers=True
        )

        # 🔍 Display the interactive plot
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_document_years")

    else:
        st.warning(f"⚠️ {warning_message}")

def display_link_node_search(link_attributes_df):
    
    # Multi-select for filtering by 'Node'
    selected_nodes = st.multiselect("Select Nodes", options=link_attributes_df["node"].unique(), default=[])

    # If nothing is selected, show full DataFrame; otherwise, filter by selected nodes
    filtered_df = link_attributes_df if not selected_nodes else link_attributes_df[link_attributes_df["node"].isin(selected_nodes)]

    # Multi-select for filtering by other attributes (excluding 'Node')
    filterable_columns = [col for col in link_attributes_df.columns if col != "node"]
    selected_columns = st.multiselect("Filter by columns", options=filterable_columns, default=[])

    # Apply column-based filtering if any attribute is selected
    for column in selected_columns:
        unique_values = filtered_df[column].unique()
        selected_values = st.multiselect(f"Select values for {column}", unique_values, default=[])
        if selected_values:  # Only filter if values are selected
            filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

    # Display the filtered DataFrame
    st.dataframe(filtered_df, use_container_width=True)
    

def display_attributes(attribute_list, key=""):
    counter = Counter(attribute_list)
    df_counts = pd.DataFrame(counter.items(), columns=["Item", "Count"]).sort_values(by="Count", ascending=False)

    # Streamlit App
    st.subheader("Pie Chart for Attribute Distribution")

    # Plot Interactive Pie Chart using Plotly
    fig = px.pie(df_counts, names="Item", values="Count",
                hover_data=["Count"], labels={"Item": "Element", "Count": "Frequency"},
                hole=0.3)  # Creates a donut-style pie chart

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_attributes_pie")

    # Show the data table for reference
    st.subheader("Count Data Table")
    st.dataframe(df_counts, use_container_width=True)

def display_dataframe(
    directory,
    suffix,
    title,
    columns=["word", "tf", "df"],
    ends=True,
    warning_message="CSV not found!",
    index_filter=None,
):
    """
    Displays a CSV as a Streamlit dataframe.

    Parameters
    ----------
    directory : str
        Folder in which to look for the CSV.
    suffix : str
        Suffix (or pattern) used by `find_by_suffix` to pick the file.
    title : str
        Sub-header shown above the dataframe.
    columns : list or None
        Which columns to keep.  None = show all columns.
    ends : bool
        Passed straight through to `find_by_suffix`.
    warning_message : str
        Message shown if the CSV cannot be found.
    index_filter : iterable or None
        Iterable of row-index values to keep; when None, the full file is shown.
    """
    file = find_by_suffix(directory, suffix=suffix, ends=ends)

    if not file:
        st.warning(f"⚠️ {warning_message}")
        return

    st.subheader(title)

    # ------------------ load ------------------
    df = pd.read_csv(file)
    if columns is not None:
        df = df[columns]

    # ------------------ filter rows by index ------------------
    if index_filter is not None:
        # Ensure we’re working with a set for fast lookup
        wanted = set(index_filter)
        df = df.loc[df.index.intersection(wanted)]

    # ------------------ display ------------------
    if df.empty:
        st.info("No rows match the selected index filter.")
    else:
        st.dataframe(df)

def display_wordcloud(directory):
    """Helper function to display a word cloud if the file exists."""
    file = find_by_suffix(directory, suffix="wordcloud_", ends=False)
    if file:
        with st_normal():  # Assuming this is a custom context manager for normalizing UI
            st.subheader("Wordcloud")
            st.image(Image.open(file), caption="Word Cloud", use_container_width=True)

def open_explorer_button(directory, key):
    """Displays a button to open the file explorer for a given directory."""
    if st.button("Open File Explorer", key=key):
        open_file_browser(directory)

def display_images_in_tabs(tab_objects, png_files):
    """Displays PNG images in corresponding tabs."""
    for j, png_file in enumerate(png_files):
        with tab_objects[j]:  
            st.write(f"🖼️ {os.path.basename(png_file)}")
            with st_normal():
                st.image(Image.open(png_file), caption=os.path.basename(png_file), use_container_width=True)

def display_html_selector(tab, html_files, key):
    """Displays a select box for choosing an HTML interactive plot."""
    with tab:
        if html_files:
            options = ["Select a file..."] + [os.path.basename(f) for f in html_files]
            selected_html = st.selectbox("Select an interactive plot:", options, key=key)
            if selected_html != "Select a file...":
                st.session_state.html_figure = next(f for f in html_files if os.path.basename(f) == selected_html)

def display_csv_in_tab(tab, directory, suffix, warning_message, ends):
    """Displays a CSV file if found, otherwise shows a warning."""
    with tab:
        file = find_by_suffix(directory, suffix=suffix, ends=ends)
        if file:
            st.dataframe(pd.read_csv(file))
        else:
            st.warning(f"⚠️ {warning_message}")

def display_view_type_setting(configuration=None):
    options = ["Document Analysis", "Link Prediction"]
    index = 0
    if configuration is not None:
        index = options.index(configuration["VIEW_TYPE"])
    st.session_state.view_type = st.selectbox("View Type:", options=options, index=index)

def display_file_node_settings(configuration=None):
    text_input_columns = st.columns(2)

    if configuration is not None:
        document_file_name_suffix = configuration.get('DOCUMENT_FILE_START_WITH') or "best_50_docs_in_"
        root_name = configuration.get('ROOT_NAME') or "*"

    else:    
        document_file_name_suffix = "best_50_docs_in_"
        root_name = "*"

    with text_input_columns[0]:
        st.session_state.document_file_name_suffix = st.text_input("Document File Start With:", document_file_name_suffix) 
    with text_input_columns[1]:
        st.session_state.root_name = st.text_input("Root Name:", root_name)         

def display_attribute_text_settings(configuration=None):
    attribute_text_columns = st.columns(3)

    if configuration is not None:
        TEXT_COLUMN = configuration.get('TEXT_COLUMN')
        ATTRIBUTE_COLUMN = configuration.get('ATTRIBUTE_COLUMN')

        if TEXT_COLUMN is not None:
            text_column_index = st.session_state.available_columns.index(TEXT_COLUMN)
        if ATTRIBUTE_COLUMN is not None:
            attribute_column_index = st.session_state.available_columns.index(ATTRIBUTE_COLUMN)

        attribute_column_split = configuration.get('ATTRIBUTE_COLUMN_SPLIT_BY') or ", "

    else:    
        text_column_index = 0
        attribute_column_index = 0
        attribute_column_split = ", "

    with attribute_text_columns[0]:
        st.session_state.text_column = st.selectbox("Text Column:", options=st.session_state.available_columns, index=text_column_index) 
    with attribute_text_columns[1]:
        st.session_state.attribute_column = st.selectbox("Attribute Column:",  options=st.session_state.available_columns, index=attribute_column_index) 
    with attribute_text_columns[2]:
        st.session_state.attribute_column_split = st.text_input("Attribute Column Split by:", attribute_column_split) 
