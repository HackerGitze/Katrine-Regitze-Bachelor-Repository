# scripts/famd_pipeline.py
import pandas as pd
import prince
import numpy as np
import os
import joblib
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

def run_famd(input_path, output_path, famd_obj_path, n_components, max_components=200, contributions_path=None, seed=None):
    """
    Run FAMD (Factor Analysis of Mixed Data) on input features.
    
    Args:
        input_path: Path to parquet file with features
        output_path: Path to save FAMD-transformed features
        famd_obj_path: Path to save the FAMD object
        n_components: Number of components to use
        max_components: Safety upper bound on components
        contributions_path: Path to save variable contributions (optional)
    
    Returns:
        Dict containing:
            - X_famd: Transformed dataframe
            - famd: Fitted FAMD object
            - contributions: DataFrame of variable contributions to components
    """

    # -------------------------------
    # Load features
    # -------------------------------

    X = pd.read_parquet(input_path)
    logger.info(f"FAMD input shape: {X.shape}")

    # Defragment DataFrame
    try:
        X = X.copy()
        logger.debug("DataFrame defragmented via X = X.copy() to improve performance")
    except Exception:
        logger.warning("Could not create defragmented copy of DataFrame; proceeding with original frame")

    # Ensure categorical columns are string dtype for FAMD fit/transform
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if len(cat_cols) > 0:
        X_for_fit = X.copy()
        for c in cat_cols:
            X_for_fit[c] = X_for_fit[c].astype(str)
        logger.debug(f"Cast {len(cat_cols)} categorical columns to string for FAMD fit/transform")
    else:
        X_for_fit = X

    # Log which columns will be treated as numeric vs categorical by FAMD.
    # Prince treats numeric dtypes as quantitative and non-numeric (object/category)
    # as categorical, so casting category->object preserves categorical semantics.
    try:
        sample_cat = cat_cols[:10]
        sample_num = numeric_cols[:10]
        logger.info(f"FAMD will treat {len(numeric_cols)} numeric columns (examples: {sample_num}) and {len(cat_cols)} categorical columns (examples: {sample_cat})")
    except Exception:
        logger.debug("Could not log column examples for FAMD detection")

    # -------------------------------
    # Safety bounding of components
    # -------------------------------
    requested = int(n_components)
    safe_max = min(max_components, X.shape[1] - 1)
    final_n_components = min(requested, safe_max)
    final_n_components = max(2, final_n_components)

    logger.info(f"Using n_components = {final_n_components} (requested: {requested})")

    # -------------------------------
    # Enable BLAS/LAPACK parallelism for faster matrix operations
    # -------------------------------
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(n_cpus)
    os.environ['MKL_NUM_THREADS'] = str(n_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
    logger.info(f"Enabled BLAS/LAPACK parallelism with {n_cpus} threads")
    
    # -------------------------------
    # Fit FAMD
    # -------------------------------
    famd = prince.FAMD(
        n_components=final_n_components,
        n_iter=10,
        random_state=(int(seed) if seed is not None else 42)
    )

    famd = famd.fit(X_for_fit)

    # -------------------------------
    # Transform
    # -------------------------------
    X_famd = famd.transform(X_for_fit)
    X_famd.columns = [f"FAMD_{i+1}" for i in range(X_famd.shape[1])]
    logger.info(f"FAMD output shape: {X_famd.shape}")

    # -------------------------------
    # Extract variable contributions
    # -------------------------------
    contributions = famd.column_contributions_
    logger.info(f"Contributions shape: {contributions.shape}")
    
    # Get top contributing variables for each component
    top_contributions = {}
    for component in contributions.columns:
        top_vars = contributions[component].nlargest(5)
        top_contributions[f"{component}"] = top_vars.to_dict()
        logger.info(f"{component}: {', '.join([f'{var}={val:.3f}' for var, val in top_vars.items()])}")

    # -------------------------------
    # Save outputs
    # -------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_famd.to_parquet(output_path)
    logger.info(f"Saved FAMD features to {output_path}")

    os.makedirs(os.path.dirname(famd_obj_path), exist_ok=True)
    joblib.dump(famd, famd_obj_path)
    logger.info(f"Saved FAMD object to {famd_obj_path}")
    
    # Save contributions if path provided
    if contributions_path:
        os.makedirs(os.path.dirname(contributions_path), exist_ok=True)
        contributions.to_parquet(contributions_path)
        logger.info(f"Saved variable contributions to {contributions_path}")

    return {
        'X_famd': X_famd,
        'famd': famd,
        'contributions': contributions,
        'top_contributions': top_contributions
    }


def run_famd_with_seeds(input_path, n_components, seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), max_components=200, top_k=10, contributions_aggregate_path=None, n_jobs=-1):
    """
    Run FAMD multiple times with different random seeds and aggregate variable contributions.

    Args:
        input_path: parquet path with input features
        n_components: requested number of components
        seeds: iterable of random seeds to try
        max_components: safety upper bound on components
        top_k: how many top variables per component to count for stability
        contributions_aggregate_path: optional path to write mean contributions (parquet)
        n_jobs: number of parallel jobs (-1 uses all CPUs, default: -1)

    Returns:
        dict with:
            - mean_contributions: DataFrame (variables x components) averaged across seeds
            - top_k_counts: dict(component -> Series) counting how often a variable appears in top_k
            - per_seed_contributions: list of DataFrames per seed
            - seeds: list of seeds used
    """

    # Enable BLAS/LAPACK parallelism for faster matrix operations
    import os as os_module
    if n_jobs == -1:
        import multiprocessing
        n_cpus = str(multiprocessing.cpu_count())
        os_module.environ['OMP_NUM_THREADS'] = n_cpus
        os_module.environ['MKL_NUM_THREADS'] = n_cpus
        os_module.environ['OPENBLAS_NUM_THREADS'] = n_cpus
        logger.info(f"Enabled BLAS/LAPACK parallelism with {n_cpus} threads")
    
    X = pd.read_parquet(input_path)
    # Ensure categorical columns are string dtype for FAMD fit/transform
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if len(cat_cols) > 0:
        X_for_fit = X.copy()
        for c in cat_cols:
            X_for_fit[c] = X_for_fit[c].astype(str)
        logger.debug(f"Cast {len(cat_cols)} categorical columns to string for FAMD fit/transform (multi-seed)")
    else:
        X_for_fit = X

    requested = int(n_components)
    safe_max = min(max_components, X.shape[1] - 1)
    final_n_components = min(requested, safe_max)
    final_n_components = max(2, final_n_components)

    top_counts = {f"Dim_{i+1}": {} for i in range(final_n_components)}

    def fit_famd_for_seed(seed):
        logger.info(f"Fitting FAMD with seed={seed}")
        famd = prince.FAMD(n_components=final_n_components, n_iter=10, random_state=int(seed))
        famd = famd.fit(X_for_fit)
        return famd.column_contributions_

    per_seed = Parallel(n_jobs=n_jobs)(delayed(fit_famd_for_seed)(seed) for seed in seeds)

    # Align components across seeds relative to the first seed.
    # We compute pairwise correlation between component contribution vectors
    # and solve an assignment problem to best-match components.
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        linear_sum_assignment = None
        logger.warning("scipy not available; component alignment will be skipped. Install scipy for robust alignment.")

    aligned_per_seed = []
    # Standard target column names
    target_cols = [f"Dim_{i+1}" for i in range(final_n_components)]

    # base contributions (first seed) and rename its columns to target names
    base = per_seed[0].copy()
    base.columns = target_cols
    aligned_per_seed.append(base)

    for idx in range(1, len(per_seed)):
        cur = per_seed[idx].copy()
        # compute correlation matrix between base cols and current cols
        if linear_sum_assignment is not None:
            corr = pd.DataFrame(index=target_cols, columns=cur.columns, dtype=float)
            for tcol, base_col in zip(target_cols, base.columns):
                for ccol in cur.columns:
                    try:
                        corr.loc[tcol, ccol] = float(base[base_col].corr(cur[ccol]))
                    except Exception:
                        corr.loc[tcol, ccol] = 0.0
            # maximize absolute correlation -> minimize negative absolute correlation
            cost = -corr.abs().fillna(0.0).values
            row_ind, col_ind = linear_sum_assignment(cost)
            # create ordered columns according to assignment
            ordered_cols = [cur.columns[c] for c in col_ind]
            cur_ordered = cur[ordered_cols]
            # rename to target names
            cur_ordered.columns = target_cols
            aligned_per_seed.append(cur_ordered)
        else:
            # fallback: just rename columns in order (may be permuted)
            cur.columns = target_cols
            aligned_per_seed.append(cur)

    # Now compute mean contributions across aligned seeds
    concat = pd.concat(aligned_per_seed, keys=[str(s) for s in seeds], names=["seed", "variable"])
    mean_contrib = concat.groupby(level=1).mean()

    # Recompute top_k counts using aligned component names
    for comp in mean_contrib.columns:
        # comp is like 'Dim_1', compute top_k on mean_contrib
        top_vars = mean_contrib[comp].nlargest(top_k).index.tolist()
        for v in top_vars:
            top_counts.setdefault(comp, {})
            top_counts[comp][v] = top_counts[comp].get(v, 0) + 1

    top_k_counts = {comp: pd.Series(counts).sort_values(ascending=False) for comp, counts in top_counts.items()}

    if contributions_aggregate_path is not None:
        os.makedirs(os.path.dirname(contributions_aggregate_path), exist_ok=True)
        mean_contrib.to_parquet(contributions_aggregate_path)
        logger.info(f"Saved mean contributions to {contributions_aggregate_path}")

    return {
        'mean_contributions': mean_contrib,
        'top_k_counts': top_k_counts,
        'per_seed_contributions': aligned_per_seed,
        'seeds': list(seeds)
    }


