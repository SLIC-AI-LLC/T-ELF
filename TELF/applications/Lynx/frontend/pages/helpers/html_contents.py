import pycountry
import matplotlib.cm as cm
import hashlib
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

DARK_COLOR = "#3b3b3b"
LIGHT_COLOR = "#E1E1E1"

def get_color_for_item(item, settings):
    stem = stemmer.stem(item.lower())

    # Handle no-color setting using visibility mode
    if settings.get("contents_coloring") == "No Color":
        visibility = settings.get("selected_visibility", "Light")
        return DARK_COLOR if visibility == "Dark" else LIGHT_COLOR

    # Otherwise, generate color from colormap
    hash_val = int(hashlib.sha256(stem.encode('utf-8')).hexdigest(), 16)
    norm_index = (hash_val % (10**8)) / (10**8)

    cmap = cm.get_cmap(settings["contents_coloring"])
    rgba = cmap(norm_index)
    rgb = tuple(int(x * 255) for x in rgba[:3])

    return f'rgb{rgb}'

def get_flag_for_country(name):
    try:
        country = pycountry.countries.get(name=name)
        if not country or len(country.alpha_2) != 2:
            return name
        code = country.alpha_2.upper()
        flag = chr(127397 + ord(code[0])) + chr(127397 + ord(code[1]))
        return f"{flag} {name}"
    except:
        return name

def render_tree_helper(data, data_map, settings):
    html = "<ul class='tree'>"
    content_map = {
        "Keyword": "unigrams", 
        "Denovo": "denovo_unigrams", 
        "Author": "author", 
        "Affiliation": "affiliation", 
        "Country": "country", 
        "Attribute": "attributes"
    }

    for node in data:
        is_expandable = bool(node["children"])
        arrow = "▶" if is_expandable else "•"
        label_class = "label expandable" if is_expandable else "label"
        # open_file_browser(data_map[node["value"]]["path"])
        # Collect additional info as separate rows
        additional_content = ""
        for sc in settings["selected_contents"]:
            content_key = content_map.get(sc)
            if content_key in data_map[node["value"]]:
                values = data_map[node["value"]][content_key]
                if settings["top_n_content_shown"] != "All":
                    values = values[:int(settings["top_n_content_shown"])]

                if sc == "Country":
                    # Choose gray tone based on theme
                    country_color = DARK_COLOR if settings.get("selected_visibility") == "Dark" else LIGHT_COLOR
                    value = ", ".join(
                        f'<span style="color: {country_color};">{get_flag_for_country(v)}</span>'
                        for v in values
                    )
                else:
                    value = ", ".join(
                        f'<span style="color: {get_color_for_item(v, settings=settings)};">{v}</span>'
                        for v in values
                    )

                additional_content += f'<div class="extra-line">{sc}: {value}</div>'

        # Apply bold only if expandable
        label_text = f"<strong>{node['label']}</strong>" if is_expandable else node['label']

        label_html = f'''
        <span class="{label_class}" onclick="toggle(this)">
            <span class="arrow">{arrow}</span> {label_text}
            {additional_content}
        </span>
        '''

        html += f"<li>{label_html}"

        if is_expandable:
            html += f'<div class="children hidden">{render_tree_helper(node["children"], data_map, settings)}</div>'

        html += "</li>"

    html += "</ul>"
    return html



def render_tree(tree_data, data_map, settings):
    # Set font color based on theme 
    font_color = "#000000" if settings["selected_visibility"] == "Dark" else "#ffffff"

    html_content = f"""
    <html>
    <head>
    <style>
    .tree {{
        list-style: none;
        padding-left: 20px;
        font-family: Arial, sans-serif;
        font-size: 15px;
    }}

    .tree li {{
        margin: 5px 0;
    }}

    .label-container {{
        cursor: default;
        user-select: none;
        display: block;
        padding: 2px 4px;
        border-radius: 4px;
    }}

    .label-container:hover {{
        background-color: rgba(200, 200, 200, 0.2);
    }}

    .label {{
        color: {font_color};
        display: inline;
    }}

    .label.expandable {{
        cursor: pointer;
        font-weight: 600;
    }}

    .arrow {{
        display: inline-block;
        width: 1em;
        transition: transform 0.2s ease;
        margin-right: 6px;
    }}

    .children {{
        margin-left: 1em;
        transition: max-height 0.3s ease-out;
    }}

    .hidden {{
        display: none;
    }}

    .extra-line {{
        display: block;
        font-weight: normal;
        font-size: 13px;
        color: #666;
        margin-left: 1.5em;
        margin-top: 2px;
    }}

    .controls button {{
        margin-right: 10px;
        padding: 6px 16px;
        font-size: 14px;
        font-weight: 500;
        background-color: #f5f5f5;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.05);
    }}

    .controls button:hover {{
        background-color: #e0e0e0;
        border-color: #bbb;
        box-shadow: 1px 2px 5px rgba(0, 0, 0, 0.1);
    }}

    </style>
    <script>
    function toggle(element) {{
        let arrow = element.querySelector('.arrow');
        let children = element.nextElementSibling;

        if (children.classList.contains('hidden')) {{
            children.classList.remove('hidden');
            arrow.style.transform = 'rotate(90deg)';
        }} else {{
            children.classList.add('hidden');
            arrow.style.transform = 'rotate(0deg)';
        }}
    }}

    function expandAll() {{
        document.querySelectorAll('.children').forEach(child => {{
            child.classList.remove('hidden');
        }});
        document.querySelectorAll('.arrow').forEach(arrow => {{
            arrow.style.transform = 'rotate(90deg)';
        }});
    }}

    function collapseAll() {{
        document.querySelectorAll('.children').forEach(child => {{
            child.classList.add('hidden');
        }});
        document.querySelectorAll('.arrow').forEach(arrow => {{
            arrow.style.transform = 'rotate(0deg)';
        }});
    }}
    </script>
    </head>
    <body>

    <div class="controls">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
    </div>

    {render_tree_helper(
        data=tree_data,
        data_map=data_map,
        settings=settings,
    )}

    </body>
    </html>
    """
    return html_content

