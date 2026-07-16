"""
Shared utility functions for the forecasting pipeline.
Centralises seed setting, column discovery, sequence building, and data splitting
to eliminate the 4× duplication across phase files.
"""
import random
import logging
import numpy as np
import pandas as pd
import torch
import os

import config

logging.basicConfig(level=logging.INFO, format='%(message)s')


def set_seed(seed=None):
    """Set random seed across all libraries for reproducibility."""
    seed = seed if seed is not None else config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")


def find_columns(df):
    """
    Discover standard column names in the dataset.
    Returns a dict with keys: target, date, category, region
    """
    # Target
    target_col = None
    for col in ['Sales', 'Sales per customer', 'Order Item Quantity']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    # Date
    date_col = next((c for c in df.columns if 'date' in c.lower() and 'order' in c.lower()), None)
    if not date_col:
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)

    # Category
    cat_col = ('Category Name' if 'Category Name' in df.columns
               else next((c for c in df.columns if 'category' in c.lower()), None))

    # Region
    region_col = ('Order Region' if 'Order Region' in df.columns
                  else next((c for c in df.columns if 'region' in c.lower()), None))

    return {
        'target': target_col,
        'date': date_col,
        'category': cat_col,
        'region': region_col,
    }


def get_feature_cols(df, cols_dict):
    """Return list of numeric feature columns, excluding target/group/date/YearWeek."""
    from pandas.api.types import is_numeric_dtype
    ignore = [cols_dict['target'], cols_dict['category'],
              cols_dict['region'], cols_dict['date'], 'YearWeek']
    return [c for c in df.columns if c not in ignore and is_numeric_dtype(df[c])]


def build_sequences(group_df, feature_cols, target_col, seq_len,
                    padding_strategy="pad"):
    """
    Slide a window of length `seq_len` over a single time-series group.
    Returns X (seq arrays), y (targets), and tab (last-timestep tabular features).
    """
    vals = group_df[feature_cols].values
    targets = group_df[target_col].values

    if len(vals) < seq_len:
        if padding_strategy == 'exclude':
            return np.array([]), np.array([]), np.array([])
        pad_len = seq_len - len(vals)
        vals = np.vstack([np.full((pad_len, vals.shape[1]), 0.0), vals])
        targets = np.concatenate([np.full((pad_len,), 0.0), targets])

    X, y, tab = [], [], []
    for i in range(len(vals) - seq_len):
        X.append(vals[i:i + seq_len])
        y.append(targets[i + seq_len])
        tab.append(vals[i + seq_len - 1])

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
    return np.array(X), np.array(y), np.array(tab)


def load_and_build_data(data_path=None, seq_len=None):
    """
    Full data-loading pipeline: read engineered CSV → discover columns →
    group by (category, region) → build sliding-window sequences →
    sort temporally → return arrays.

    Returns: X_seq, y_seq, X_tab, feature_cols, cols_dict
    """
    data_path = data_path or config.ENGINEERED_DATA_PATH
    if not os.path.exists(data_path):
        logging.error(f"[ERROR] Dataset '{data_path}' not found.")
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    cols = find_columns(df)

    if cols['date']:
        df = df.sort_values(cols['date'])

    feature_cols = get_feature_cols(df, cols)

    if seq_len is None:
        seq_len = config.load_best_seq_len()

    cat_col = cols['category']
    region_col = cols['region']

    if not cat_col or not region_col:
        raise ValueError("Category and Region columns are required for grouping.")

    X_seq_all, y_seq_all, X_tab_all, dates_all = [], [], [], []

    for _, group in df.groupby([cat_col, region_col]):
        group = group.sort_values(cols['date']) if cols['date'] else group
        X, y, tab = build_sequences(group, feature_cols, cols['target'],
                                    seq_len, config.PADDING_STRATEGY)
        if len(X) > 0:
            X_seq_all.append(X)
            y_seq_all.append(y)
            X_tab_all.append(tab)
            dates = group[cols['date']].values if cols['date'] else np.arange(len(group))
            # Align dates with targets (skip first seq_len of padded series)
            padded_len = len(group[feature_cols].values)
            if padded_len < seq_len:
                padded_len = seq_len  # after padding
            actual_dates = group[cols['date']].values if cols['date'] else np.arange(len(group))
            # Get dates corresponding to each target
            if len(actual_dates) < seq_len:
                pad_len = seq_len - len(actual_dates)
                actual_dates = np.concatenate([np.full((pad_len,), actual_dates[0]), actual_dates])
            for i in range(len(actual_dates) - seq_len):
                dates_all.append(actual_dates[i + seq_len])

    if not X_seq_all:
        raise ValueError("No sequences could be built from the data.")

    X_seq_all = np.concatenate(X_seq_all)
    y_seq_all = np.concatenate(y_seq_all)
    X_tab_all = np.concatenate(X_tab_all)
    dates_all = np.array(dates_all)

    # Sort by date
    sort_idx = np.argsort(dates_all)
    X_seq_all = X_seq_all[sort_idx]
    y_seq_all = y_seq_all[sort_idx]
    X_tab_all = X_tab_all[sort_idx]

    return X_seq_all, y_seq_all, X_tab_all, feature_cols, cols


def temporal_train_val_test_split(X_seq, y_seq, X_tab=None):
    """
    Three-way temporal split (70/15/15).
    Returns dict with keys 'train', 'val', 'test', each containing
    'X_seq', 'y', and optionally 'X_tab'.
    """
    n = len(X_seq)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    splits = {
        'train': {
            'X_seq': X_seq[:train_end],
            'y': y_seq[:train_end],
        },
        'val': {
            'X_seq': X_seq[train_end:val_end],
            'y': y_seq[train_end:val_end],
        },
        'test': {
            'X_seq': X_seq[val_end:],
            'y': y_seq[val_end:],
        },
    }

    if X_tab is not None:
        splits['train']['X_tab'] = X_tab[:train_end]
        splits['val']['X_tab'] = X_tab[train_end:val_end]
        splits['test']['X_tab'] = X_tab[val_end:]

    logging.info(f"Split sizes — Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
    return splits
