# merge_features.py
import argparse
from pathlib import Path
import pandas as pd

def norm_id_col(df):
    # try to find id column and normalize its name to "id"
    for cand in ("ID","Id","id","sample_id","sample","filename"):
        if cand in df.columns:
            df = df.rename(columns={cand: "id"})
            return df
    # fallback: first column as id
    df = df.rename(columns={df.columns[0]: "id"})
    return df

def strip_ext_from_id(df, id_col="id"):
    # if ids contain extensions, strip them
    df[id_col] = df[id_col].astype(str).apply(lambda x: Path(x).stem)
    return df

def main(args):
    hu = Path(args.hu_csv) if args.hu_csv else None
    pca = Path(args.pca_csv) if args.pca_csv else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    if hu and hu.exists():
        df_hu = pd.read_csv(str(hu), dtype=str)
        df_hu = norm_id_col(df_hu)
        df_hu = strip_ext_from_id(df_hu)
        dfs.append(df_hu)
        print(f"Loaded HU: {hu} ({len(df_hu)} rows)")
    if pca and pca.exists():
        df_pca = pd.read_csv(str(pca), dtype=str)
        df_pca = norm_id_col(df_pca)
        df_pca = strip_ext_from_id(df_pca)
        dfs.append(df_pca)
        print(f"Loaded PCA: {pca} ({len(df_pca)} rows)")

    if not dfs:
        raise SystemExit("No input CSVs found. Provide at least one of --hu or --pca")

    # Merge on 'id' using outer join to keep all ids
    df_merged = dfs[0]
    for other in dfs[1:]:
        df_merged = df_merged.merge(other, on="id", how="outer", suffixes=("", "_p"))

    # attempt to cast numeric columns to numeric (where possible)
    for col in df_merged.columns:
        if col == "id":
            continue
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    df_merged = df_merged.fillna(0)  # fill missing with zeros (or choose other strategy)

    df_merged.to_csv(str(out), index=False)
    print(f"Saved merged features to {out} ({len(df_merged)} rows, {len(df_merged.columns)-1} features)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hu_csv", type=str, default="src/data/processed/2d_hu/hu_features_table.csv")
    p.add_argument("--pca_csv", type=str, default="src/data/processed/descriptors/descriptors_table_pca.csv")
    p.add_argument("--out", type=str, default="src/data/features/features_table.csv")
    args = p.parse_args()
    main(args)
