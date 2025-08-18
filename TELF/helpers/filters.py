from typing import Any, Sequence
import pandas as pd
import ast
import re


def get_papers(df, authors, col):
    """
    Filters a DataFrame for papers based on the presence of a highlighted author from authors
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to be filtered.
    authors: set
        A set or any iterable containing author id strings to search for in the DataFrame. If the provided 
        selection is not a set, it will be converted to one.
    col: str
        The column name in the DataFrame where author ids from `authors` should be searched.
    
    Returns:
    --------
    pd.DataFrame: 
        A subset of the original DataFrame containing rows where any highlighted author from authors 
        is found in the specified column's values.
    """
    if not isinstance(authors, set):
        selection = set(authors)
    def is_in_set(s):
        return any(x in authors for x in s.split(';'))
    return df[df[col].apply(is_in_set)].copy()

def filter_author_map(df, *, country=None, affiliation=None):
    """
    Filter a SLIC author map by country, affiliation id, or both
    
    Parameters:
    -----------
    df: pd.Data
        The SLIC author map being filtered
    country: str
        Some country by which to filter.
    affiliation: str
        The Scopus affiliation by which to filter
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    ids = set(df.index.to_list())
    for index, row in df.iterrows():
        affiliations = row.scopus_affiliations
        if pd.isna(affiliations):
            ids.remove(index)
            continue
            
        if isinstance(affiliations, str):
            affiliations = ast.literal_eval(affiliations)
        
        countries = {x['country'].lower() for x in affiliations.values()}
        matched = False
        for c in countries:
            if country.lower() in c:  # substring comparison
                matched = True
        if not matched:
            ids.remove(index)
        
        affiliation_ids = set(affiliations.keys())
        if affiliation is not None and affiliation not in affiliation_ids:
            ids.remove(index)
    
    return df.iloc[list(ids)]


def find_subdf(df: pd.DataFrame, column: str, ids: Sequence[str]) -> pd.DataFrame:
    """
    Filter a DataFrame for rows where a column contains any of the specified IDs
    as semicolon‐separated tokens.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Name of the column in `df` containing semicolon‐separated identifiers.
    ids : sequence of str
        List of identifier strings to match as whole tokens.

    Returns
    -------
    pandas.DataFrame
        Subset of `df` containing only rows where `column` contains any of the
        specified `ids` (delimited by semicolons or string boundaries).
    """
    escaped_ids = [re.escape(id_) for id_ in ids]
    pattern = r'(?:^|;\s*)(' + '|'.join(escaped_ids) + r')(?:\s*;|$)'
    mask = df[column].str.contains(pattern, regex=True)
    return df[mask]


def remove_unknown_entries(cell: Any) -> Any:
    """
    Remove dictionary‐like 'Unknown' entries from a string, cleaning up commas.

    This function removes any substring matching
    `[, ]?'Unknown': [ ... ]` and then tidies up leftover commas.

    Parameters
    ----------
    cell : any
        If not a string, returned unchanged. If a string, cleaned.

    Returns
    -------
    any
        The cleaned string with 'Unknown' entries removed, or the original
        non‐string `cell`.
    """
    if not isinstance(cell, str):
        return cell

    s = cell
    # Remove "'Unknown': [...]" entries (with optional leading comma)
    s = re.sub(r",?\s*'Unknown'\s*:\s*\[[^\]]*\]", "", s)
    # Clean up trailing commas before braces or ends
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r"^,\s*", "", s)
    s = re.sub(r",\s*$", "", s)
    return s.strip()





def normalize_affiliation_cell(cell):
    """
    Turn whatever is in `cell` into a dict of affiliation‐info dicts.
    - If it’s already a dict, return as-is.
    - If it’s a list, enumerate it into a dict.
    - If it’s a string, optionally clean out Unknowns, then ast.literal_eval it.
    - Otherwise return {}.
    """
    if isinstance(cell, dict):
        return cell

    if isinstance(cell, list):
        # turn [info0, info1, …] into {0:info0, 1:info1, …}
        return {i: entry for i, entry in enumerate(cell)}

    if isinstance(cell, str):
        # 1) strip out 'Unknown':[…] patterns
        s = remove_unknown_entries(cell)
        # 2) now parse it safely back into a Python object
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return {}
        # 3) if the result is a list or dict, normalize it:
        return normalize_affiliation_cell(parsed)

    # anything else → no affiliations
    return {}

def clean_affiliations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'affiliations' column by removing unknown entries in each cell.

    Applies `remove_unknown_entries` to every entry of the 'affiliations' column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with an 'affiliations' column.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with 'affiliations' cleaned in‐place.
    """
    df['affiliations'] = df['affiliations'].apply(remove_unknown_entries)
    df['affiliations'] = df['affiliations'].apply(normalize_affiliation_cell)
    
    return df
