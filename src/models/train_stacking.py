#!/usr/bin/env python3
"""Train a multi-view stacking classifier on Hu and PCA features.
Examples:
    # Full run
    python train_stacking.py --hu_csv ../../data/processed/2d_hu/hu_features_table.csv --pca_csv ../../data/processed/descriptors/descriptors_table_pca.csv  --test_csv ../../data/raw/test.csv --out_dir ../../experiments --binarize median --meta_clf logistic

    # Fast (debug) run
    python train_stacking.py --hu_csv ../../data/processed/2d_hu/hu_features_table.csv --pca_csv ../../data/processed/descriptors/descriptors_table_pca.csv  --test_csv ../../data/raw/test.csv --out_dir experiments --binarize otsu --meta_clf xgboost --fast
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from skimage.filters import threshold_otsu


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-view stacking with Hu and PCA features.')
    parser.add_argument('--hu_csv', type=str, required=True, help='../../experiments')
    parser.add_argument('--pca_csv', type=str, required=True, help='Path to descriptors_table_pca.csv')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv (IDs to exclude)')
    parser.add_argument('--out_dir', type=str, default='experiments', help='Output base directory')
    parser.add_argument('--binarize', type=str, default='median',
                        choices=['median', 'quantile', 'otsu'], help='Binarization method')
    parser.add_argument('--meta_clf', type=str, default='logistic',
                        choices=['logistic', 'xgboost'], help='Meta-classifier')
    parser.add_argument('--fast', action='store_true', help='Use debug mode with limited samples')
    parser.add_argument('--verbose', action='store_true', help='Verbose output (debug logging)')
    return parser.parse_args()


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def load_data(hu_path, pca_path, test_path):
    logging.info('Loading data...')
    hu_df = pd.read_csv(hu_path)
    pca_df = pd.read_csv(pca_path)
    test_df = pd.read_csv(test_path)
    # Remove unwanted columns (raw_path not needed)
    if 'raw_path' in pca_df.columns:
        pca_df = pca_df.drop(columns=['raw_path'])
    return hu_df, pca_df, test_df


def extract_test_ids(test_df):
    # Extract IDs from test paths: species_firstNumber
    test_ids = set()
    for path in test_df['path']:
        if pd.isna(path):
            continue
        parts = path.split('/', 1)
        if len(parts) != 2:
            continue
        species, rest = parts
        import re
        m = re.search(r'\d+', rest)
        if m:
            num = m.group(0)
            id_val = f"{species}_{num}"
            test_ids.add(id_val)
    logging.debug(f'Extracted {len(test_ids)} test IDs.')
    return test_ids


def binarize_features(X, method='median', verbose=False):
    thresholds = {}
    X_bin = pd.DataFrame(index=X.index, columns=X.columns)
    for col in X.columns:
        vals = X[col].values
        if method == 'median':
            thr = np.nanmedian(vals)
        elif method == 'quantile':
            # Default quantile at 0.5 (median)
            thr = np.nanpercentile(vals, 50)
        elif method == 'otsu':
            try:
                thr = float(threshold_otsu(np.array(vals, dtype=float)))
            except Exception:
                thr = np.nanmedian(vals)
        else:
            raise ValueError(f"Unknown binarization method: {method}")
        thresholds[col] = float(thr)
        X_bin[col] = (X[col].fillna(thr) >= thr).astype(int)
        if verbose:
            logging.debug(f"Binarize feature '{col}': threshold={thr}, method={method}")
    return X_bin, thresholds


def main():
    args = parse_args()
    setup_logging(args.verbose)
    np.random.seed(42)

    # 1. Load data
    hu_df, pca_df, test_df = load_data(args.hu_csv, args.pca_csv, args.test_csv)

    # 2. Merge on id and extract labels
    logging.info('Merging Hu and PCA feature tables on id...')
    df = pd.merge(hu_df, pca_df, on='id', how='inner')
    logging.info('Extracting labels from ids...')
    df['label'] = df['id'].apply(lambda x: x.split('_')[0])

    # 3. Exclude test IDs
    test_ids = extract_test_ids(test_df)
    if test_ids:
        logging.info(f"Excluding {len(test_ids)} IDs present in test set from data.")
        df = df[~df['id'].isin(test_ids)].reset_index(drop=True)

    # 4. Fast mode subset
    if args.fast:
        logging.info('Fast mode: using a small subset of data for quick debugging.')
        df = df.sample(n=min(20, len(df)), random_state=42).reset_index(drop=True)

    logging.info(f"Data size after filtering: {len(df)} samples.")

    # 5. Prepare feature matrices and labels
    hu_features = [c for c in hu_df.columns if c not in ['id', 'note']]
    pca_features = [c for c in pca_df.columns if c not in ['id']]

    X_hu = df[hu_features].copy()
    X_pca = df[pca_features].copy()
    y = df['label'].values

    # Encode labels globally
    le_global = LabelEncoder()
    y_enc = le_global.fit_transform(y)
    class_names = le_global.classes_
    y_enc = y_enc - y_enc.min()
    # 6. Split into train/val sets (20% val)
    indices = np.arange(len(df))
    logging.info('Splitting data into train and validation sets...')
    if args.fast:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    else:
        train_idx, val_idx = train_test_split(indices, test_size=0.2,
                                               random_state=42, stratify=y_enc)
    
    X_hu_train, X_hu_val = X_hu.iloc[train_idx], X_hu.iloc[val_idx]
    X_pca_train, X_pca_val = X_pca.iloc[train_idx], X_pca.iloc[val_idx]
    y_train, y_val = y_enc[train_idx], y_enc[val_idx]
    
    y_train = y_train - y_train.min()
    y_val   = y_val - y_val.min()


    # 7. Binarize features using chosen method
    logging.info(f'Applying binarization ({args.binarize}) to Hu features...')
    X_hu_train_bin, hu_thresholds = binarize_features(X_hu_train, method=args.binarize, verbose=args.verbose)
    X_hu_val_bin = (X_hu_val.fillna(pd.Series(hu_thresholds))
                    .ge(pd.Series(hu_thresholds))).astype(int)

    logging.info(f'Applying binarization ({args.binarize}) to PCA features...')
    X_pca_train_bin, pca_thresholds = binarize_features(X_pca_train, method=args.binarize, verbose=args.verbose)
    X_pca_val_bin = (X_pca_val.fillna(pd.Series(pca_thresholds))
                     .ge(pd.Series(pca_thresholds))).astype(int)

    # 8. Train base BernoulliNB models
    logging.info('Training BernoulliNB on Hu features...')
    nb_hu = BernoulliNB()
    nb_hu.fit(X_hu_train_bin, y_train)
    logging.info('Training BernoulliNB on PCA features...')
    nb_pca = BernoulliNB()
    nb_pca.fit(X_pca_train_bin, y_train)

    # 9. Get base predictions (probabilities) on validation set
    logging.info('Predicting probabilities with base classifiers...')
    hu_proba = nb_hu.predict_proba(X_hu_val_bin)
    pca_proba = nb_pca.predict_proba(X_pca_val_bin)

    # 10. Prepare meta features (concatenate probabilities)
    X_meta_val = np.concatenate([hu_proba, pca_proba], axis=1)

    # 11. Train meta-classifier
    logging.info(f'Training meta-classifier ({args.meta_clf})...')
    if args.meta_clf == 'logistic':
        meta_clf = LogisticRegression(random_state=42, max_iter=1000)
        meta_clf.fit(X_meta_val, y_val)
    elif args.meta_clf == 'xgboost':
        if XGBClassifier is None:
            logging.error("XGBoost is not installed, cannot use xgboost meta-clf.")
            sys.exit(1)
        meta_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        meta_clf.fit(X_meta_val, y_val)
    else:
        logging.error(f"Unsupported meta-classifier: {args.meta_clf}")
        sys.exit(1)

    # 12. Predict with stacking classifier on validation set
    logging.info('Predicting with stacking classifier on validation set...')
    y_val_pred = meta_clf.predict(X_meta_val)

    # 13. Compute metrics
    logging.info('Computing performance metrics...')
    acc = accuracy_score(y_val, y_val_pred)
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)
    f1_macro = f1_score(y_val, y_val_pred, average='macro')
    precisions = precision_score(y_val, y_val_pred, average=None, labels=range(len(class_names)))
    recalls = recall_score(y_val, y_val_pred, average=None, labels=range(len(class_names)))

    # 14. Prepare output paths
    exp_dir = os.path.join(args.out_dir, "mv_stack")
    if args.fast:
        exp_dir = os.path.join(exp_dir, "_fast_debug")
    os.makedirs(exp_dir, exist_ok=True)

    # Save binarization thresholds to YAML
    yaml_path = os.path.join(exp_dir, 'bin_thresholds.yaml')
    logging.info(f'Saving binarization thresholds to {yaml_path}')
    thresholds_out = {'hu_features': hu_thresholds, 'pca_features': pca_thresholds}
    with open(yaml_path, 'w') as f:
        yaml.dump(thresholds_out, f)

    # Save metrics to CSV
    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    logging.info(f'Saving metrics to {metrics_path}')
    metrics_data = []
    metrics_data.append({'metric': 'accuracy', 'class': '', 'value': acc})
    metrics_data.append({'metric': 'balanced_accuracy', 'class': '', 'value': bal_acc})
    metrics_data.append({'metric': 'macro_f1', 'class': '', 'value': f1_macro})
    for idx, cls in enumerate(class_names):
        metrics_data.append({'metric': 'precision', 'class': cls, 'value': precisions[idx]})
        metrics_data.append({'metric': 'recall', 'class': cls, 'value': recalls[idx]})
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_path, index=False)

    # Save predictions to CSV
    pred_path = os.path.join(exp_dir, 'predictions.csv')
    logging.info(f'Saving predictions to {pred_path}')
    df_pred = pd.DataFrame({
        'id': df.loc[val_idx, 'id'].values,
        'true': df.loc[val_idx, 'label'].values,
        'predicted': le_global.inverse_transform(y_val_pred)
    })

    df_pred.to_csv(pred_path, index=False)

    logging.info('Done.')


if __name__ == '__main__':
    main()
