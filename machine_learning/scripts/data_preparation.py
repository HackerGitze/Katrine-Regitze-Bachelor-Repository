# scripts/data_preparation.py
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(parquet_path):
    """Load the encoded dataset."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} does not exist")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded df shape: {df.shape}")
    return df

def split_features_outcome(df, outcome_var, exclude_cols=None):
    """
    Split df into X (features) and y (outcome).
    
    Args:
        df: Input dataframe
        outcome_var: Name of the outcome variable
        exclude_cols: List of columns to exclude (e.g., IDs, metadata). Defaults to ['CASEID_1979']
    """
    if outcome_var not in df.columns:
        raise ValueError(f"Outcome variable '{outcome_var}' not found in df.columns. Available: {df.columns.tolist()}")
    
    # Default exclusions: outcome + common ID columns
    if exclude_cols is None:
        exclude_cols = ['CASEID_1979']
    
    # Always include outcome in drops
    cols_to_drop = [outcome_var] + [col for col in exclude_cols if col in df.columns and col != outcome_var]
    
    # Log which ID columns are being excluded
    id_cols_found = [col for col in exclude_cols if col in df.columns]
    if id_cols_found:
        logger.info(f"Excluding ID/metadata columns: {id_cols_found}")
    
    X = df.drop(columns=cols_to_drop)
    y = df[outcome_var]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def save_features_for_famd(X, save_path):
    """Save feature dataframe for FAMD.
    
    Converts categorical columns to string dtype before saving to ensure
    consistency when FAMD transforms the data later.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert categorical columns to string for FAMD consistency
    X_save = X.copy()
    cat_cols = X_save.select_dtypes(include=["category", "object"]).columns.tolist()
    if len(cat_cols) > 0:
        for c in cat_cols:
            X_save[c] = X_save[c].astype(str)
        logger.info(f"Converted {len(cat_cols)} categorical columns to string dtype before saving for FAMD")
    
    X_save.to_parquet(save_path)
    logger.info(f"Saved features for FAMD to {save_path}")
    return save_path


def apply_encodings_and_save(interim_parquet_path, encodings_csv_path, output_parquet_path, engine="pyarrow"):
    """Load interim data + encodings, apply dtypes and save a parquet with preserved dtypes.

    This function uses the encodings CSV (must have columns `variable_name` and
    `variable_type`, optional `categories` for ordinals) and writes a parquet
    using `pyarrow` so pandas dtypes (categorical, nullable ints) are preserved.
    """
    import ast

    if not os.path.exists(interim_parquet_path):
        raise FileNotFoundError(f"Interim parquet not found: {interim_parquet_path}")
    if not os.path.exists(encodings_csv_path):
        raise FileNotFoundError(f"Encodings CSV not found: {encodings_csv_path}")

    encodings = pd.read_csv(encodings_csv_path)
    df = pd.read_parquet(interim_parquet_path)

    logger.info(f"Loaded interim data shape={df.shape}; encodings rows={len(encodings)}")

    # Safety checks
    missing_in_df = set(encodings["variable_name"]) - set(df.columns)
    missing_in_enc = set(df.columns) - set(encodings["variable_name"]) 
    if missing_in_df:
        logger.warning("Variables in encodings missing from df: %s", missing_in_df)
    if missing_in_enc:
        logger.warning("Variables in df missing from encodings: %s", list(missing_in_enc)[:10])

    # Keep only encodings for columns we actually have
    encodings = encodings[encodings["variable_name"].isin(df.columns)].copy()

    def apply_encoding(series, enc_type, encodings_df):
        name = series.name
        # NUMERIC / CONTINUOUS
        if enc_type in ["numeric", "continuous"]:
            return pd.to_numeric(series, errors="coerce").astype("float64")

        # YEAR -> integer (nullable)
        if enc_type == "year":
            return pd.to_numeric(series, errors="coerce").astype("Int64")

        # BINARY -> keep as categorical for FAMD; also preserve missing
        if enc_type == "binary":
            # Try to coerce known SAS missing codes to pd.NA
            s = pd.to_numeric(series, errors="coerce")
            missing_codes = {-1, -2, -3, -4, -5, 997, 998, 999}
            s = s.replace(list(missing_codes), pd.NA)

            # Map common encodings (1/2 -> 1/0)
            vals = set(s.dropna().unique())
            if vals.issubset({1, 2}):
                s = s.map({1: 1, 2: 0})
            # Map text yes/no
            elif any(v in {"Yes", "No", "Y", "N"} for v in series.dropna().unique()):
                return series.map({"Yes": 1, "No": 0, "Y": 1, "N": 0}).astype("category")
            # finally cast to category (keeps missing)
            return s.astype("Int8").astype("category")

        # NOMINAL categorical
        if enc_type == "categorical_nominal":
            return series.astype("category")

        # ORDINAL: respect provided order if present
        if enc_type == "categorical_ordinal":
            try:
                row = encodings_df.loc[encodings_df["variable_name"] == name]
                if "categories" in row.columns and len(row) > 0 and pd.notna(row["categories"].values[0]):
                    cats = row["categories"].values[0]
                    if isinstance(cats, str):
                        cats = ast.literal_eval(cats)
                    return pd.Categorical(series, categories=cats, ordered=True)
            except Exception as e:
                logger.warning("Could not apply ordered categories for %s: %s", name, e)
            # fallback: infer order from sorted unique values
            unique_vals = sorted(series.dropna().unique())
            return pd.Categorical(series, categories=unique_vals, ordered=True)

        raise ValueError(f"Unknown encoding type: {enc_type} for variable {name}")

    encoded = df.copy()
    for var, enc_type in encodings[["variable_name", "variable_type"]].values:
        try:
            encoded[var] = apply_encoding(encoded[var], enc_type, encodings)
        except Exception as e:
            logger.warning("Encoding failed for %s: %s", var, e)

    # --- Diagnostic metadata ---
    import json

    categories = {}
    for c, t in encoded.dtypes.items():
        if str(t).startswith("category"):
            try:
                # convert categories to native python types for JSON
                raw_cats = list(encoded[c].cat.categories)
                serializable = []
                for v in raw_cats:
                    try:
                        # numpy scalar -> python native
                        if hasattr(v, 'item'):
                            serializable.append(v.item())
                        else:
                            serializable.append(v)
                    except Exception:
                        serializable.append(str(v))
                categories[c] = serializable
            except Exception:
                categories[c] = []

    nullable_ints = [c for c, t in encoded.dtypes.items() if str(t).startswith("Int")]

    dtype_map = {c: str(t) for c, t in encoded.dtypes.items()}

    metadata = {
        "categories": categories,
        "nullable_ints": nullable_ints,
        "dtypes": dtype_map,
        "shape": list(encoded.shape)
    }

    # Save parquet using pyarrow to preserve dtypes like categories and nullable ints
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    # Prefer pyarrow for best dtype fidelity; warn if it's not available
    try:
        if engine == "pyarrow":
            import pyarrow  # noqa: F401
        encoded.to_parquet(output_parquet_path, engine=engine)
    except ImportError:
        logger.warning("pyarrow not available; writing parquet with default engine. Install pyarrow to preserve pandas dtypes (categories, nullable ints).")
        encoded.to_parquet(output_parquet_path)
    except Exception as e:
        logger.error("Failed to write parquet with engine=%s: %s", engine, e)
        # fallback without specifying engine
        encoded.to_parquet(output_parquet_path)

    # Write metadata JSON alongside parquet
    meta_path = output_parquet_path + ".metadata.json"
    try:
        with open(meta_path, "w") as mf:
            json.dump(metadata, mf, indent=2)
        logger.info("Saved metadata to %s", meta_path)
    except Exception as e:
        logger.warning("Failed to write metadata JSON: %s", e)

    # Log a short diagnostics summary for quick inspection
    cat_sample = {c: (categories.get(c, [])[:10]) for c in list(categories)[:20]}
    logger.info("Saved encoded dataset to %s (shape=%s)", output_parquet_path, encoded.shape)
    logger.info("Categorical columns detected: %s", list(categories.keys())[:50])
    logger.debug("Categories sample: %s", cat_sample)

    return output_parquet_path


def reload_with_metadata(parquet_path):
    """Load a parquet and, if present, reapply dtypes from a sidecar metadata JSON.

    The metadata JSON is expected at `<parquet_path>.metadata.json` and to
    contain the keys written by `apply_encodings_and_save`: `categories`,
    `nullable_ints`, and `dtypes`.
    Returns a DataFrame with dtypes restored where possible.
    """
    meta_path = parquet_path + ".metadata.json"
    df = pd.read_parquet(parquet_path)
    if not os.path.exists(meta_path):
        return df

    import json
    try:
        with open(meta_path, "r") as mf:
            meta = json.load(mf)
    except Exception:
        logger.warning("Could not read metadata JSON at %s; returning raw DataFrame", meta_path)
        return df

    # Reapply category dtypes (preserve order if provided)
    categories = meta.get("categories", {})
    for col, cats in categories.items():
        if col in df.columns:
            try:
                df[col] = pd.Categorical(df[col], categories=cats, ordered=False)
            except Exception as e:
                logger.debug("Failed to reapply categories for %s: %s", col, e)

    # Reapply nullable integer dtypes
    for col in meta.get("nullable_ints", []):
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            except Exception as e:
                logger.debug("Failed to reapply nullable int dtype for %s: %s", col, e)

    # Optionally reapply any explicit dtype strings recorded
    dtypes = meta.get("dtypes", {})
    for col, dt in dtypes.items():
        if col not in df.columns:
            continue
        # skip categories and nullable ints already handled
        if col in categories or col in meta.get("nullable_ints", []):
            continue
        try:
            # only attempt safe casts for common types
            if dt in ("float64", "float32"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dt in ("int64", "int32"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dt.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            logger.debug("Skipping dtype reapplication for %s -> %s", col, dt)

    logger.info("Reloaded parquet with metadata applied: %s", parquet_path)
    return df
