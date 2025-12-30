import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb
import logging
import joblib

logger = logging.getLogger(__name__)


def _plot_top_k(mean_contrib, comp, top_k=10, out_dir=None):
    """Create a horizontal bar plot for top_k variables of a single component.
    Returns the path to the saved PNG.
    """
    series = mean_contrib[comp].nlargest(top_k)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(series))))
    y = np.arange(len(series))
    ax.barh(y, series.values, color="C0")
    ax.set_yticks(y)
    ax.set_yticklabels(series.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean contribution")
    ax.set_title(f"Top {top_k} contributors — {comp}")
    plt.tight_layout()
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"famd_{comp}_top{top_k}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def log_mean_contributions_wandb(mean_contrib, project, entity=None, name="famd_mean_contributions", top_k=10, out_dir="./outputs/famd_viz"):
    """Log the mean contributions DataFrame to W&B as a Table and top-k plots per component.

    Args:
        mean_contrib: DataFrame (variables x components) with numeric contributions
        project: W&B project name
        entity: optional W&B entity/user
        name: run name for this logging run
        top_k: number of top variables per component to plot
        out_dir: local directory to save images before upload
    """
    # Ensure DataFrame shape
    if not isinstance(mean_contrib, pd.DataFrame):
        raise ValueError("mean_contrib must be a pandas DataFrame")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory for FAMD viz: {out_dir}")

    # Start a W&B run for logging this artifact
    logger.info(f"Initializing W&B run: project={project}, entity={entity}, name={name}")
    wandb.init(project=project, entity=entity, name=name, reinit=True)

    # Prepare a Table: rows are (component, variable, mean_contribution, rank)
    cols = ["component", "variable", "mean_contribution", "rank"]
    table = wandb.Table(columns=cols)
    for comp in mean_contrib.columns:
        s = mean_contrib[comp].dropna().sort_values(ascending=False)
        for rank, (var, val) in enumerate(s.items(), start=1):
            table.add_data(comp, var, float(val), int(rank))

    wandb.log({"famd_mean_contributions_table": table})
    logger.info(f"Logged W&B Table with {len(mean_contrib)} variables and {len(mean_contrib.columns)} components")

    # Create and upload top_k plots
    plot_paths = []
    for comp in mean_contrib.columns:
        try:
            p = _plot_top_k(mean_contrib, comp, top_k=top_k, out_dir=out_dir)
            plot_paths.append(p)
            logger.info(f"Saving plot for {comp} to {p}")
            wandb.log({f"famd_top_{comp}": wandb.Image(p)})
            logger.info(f"Logged plot for {comp} to W&B")
        except Exception as e:
            logger.error(f"Failed to plot component {comp}: {e}", exc_info=True)

    # Optionally create an artifact with the table and images
    logger.info(f"Creating W&B Artifact with {len(plot_paths)} plots")
    try:
        art = wandb.Artifact("famd_mean_contributions", type="dataset")
        # attach the parquet-like CSV
        csv_path = os.path.join(out_dir, "famd_mean_contributions.csv")
        mean_contrib.to_csv(csv_path)
        art.add_file(csv_path)
        logger.info(f"Added CSV to artifact: {csv_path}")
        for p in plot_paths:
            art.add_file(p)
            logger.debug(f"Added plot file to artifact: {p}")
        wandb.log_artifact(art)
        logger.info(f"Successfully logged W&B Artifact 'famd_mean_contributions'")
    except Exception as e:
        logger.error(f"Failed to create W&B artifact: {e}", exc_info=True)
        wandb.finish()
        raise

    wandb.finish()
    logger.info("Finished W&B logging for FAMD visualizations")


def build_mean_contrib_from_parquet(path):
    """Helper to load a parquet file into DataFrame for visualization."""
    return pd.read_parquet(path)


def visualize_famd(famd_obj_path, X_famd_path=None, output_dir="./outputs/famd_viz", n_components=10):
    """Create local matplotlib visualizations for a fitted FAMD object.

    Produces:
      - explained inertia bar plot
      - contribution heatmap for the first `n_components` components

    Args:
        famd_obj_path: path to saved joblib FAMD object
        X_famd_path: optional path to saved FAMD-transformed features (parquet)
        output_dir: directory to save plots
        n_components: how many components to visualize (columns)
    """
    os.makedirs(output_dir, exist_ok=True)

    famd = joblib.load(famd_obj_path)

    # Explained inertia
    try:
        explained = getattr(famd, "explained_inertia_", None)
    except Exception:
        explained = None

    if explained is not None:
        plt.figure(figsize=(8, 4))
        comps = np.arange(1, len(explained) + 1)
        plt.bar(comps, explained, color="C0")
        plt.xlabel("FAMD Component")
        plt.ylabel("Explained Inertia")
        plt.title("FAMD Component Explained Inertia")
        plt.tight_layout()
        explained_path = os.path.join(output_dir, "famd_explained_inertia.png")
        plt.savefig(explained_path, dpi=150)
        plt.close()
        logger.info(f"Saved explained inertia plot to {explained_path}")

    # Column contributions heatmap
    contribs = getattr(famd, "column_contributions_", None)
    if contribs is None:
        logger.warning("FAMD object does not expose 'column_contributions_'; skipping contribution heatmap")
        return

    # select first n_components (safely)
    n_comp = min(n_components, contribs.shape[1])
    cols = contribs.columns[:n_comp]
    sub = contribs[cols]

    # Plot heatmap via imshow for compatibility (no seaborn)
    fig, ax = plt.subplots(figsize=(max(6, n_comp * 0.6), min(12, 0.25 * sub.shape[0] + 2)))
    im = ax.imshow(sub.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(n_comp))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(sub.index)))
    # show only up to 200 yticklabels, otherwise skip labels for readability
    if len(sub.index) <= 200:
        ax.set_yticklabels(sub.index)
    else:
        ax.set_yticklabels(["" for _ in sub.index])
    ax.set_xlabel("Component")
    ax.set_ylabel("Feature")
    plt.colorbar(im, ax=ax, label="Contribution")
    plt.title(f"Top {n_comp} Component Contributions")
    plt.tight_layout()
    contrib_path = os.path.join(output_dir, f"famd_top{n_comp}_contributions.png")
    fig.savefig(contrib_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved contribution heatmap to {contrib_path}")

    # Optionally save the subset contributions CSV for inspection
    csv_out = os.path.join(output_dir, f"famd_top{n_comp}_contributions.csv")
    sub.to_csv(csv_out)
    logger.info(f"Saved contribution table to {csv_out}")

    # If X_famd provided, save a preview of the transformed features
    if X_famd_path is not None and os.path.exists(X_famd_path):
        try:
            Xf = pd.read_parquet(X_famd_path)
            preview_path = os.path.join(output_dir, "famd_features_preview.csv")
            Xf.head(200).to_csv(preview_path, index=False)
            logger.info(f"Saved preview of transformed features to {preview_path}")
        except Exception as e:
            logger.warning(f"Could not read X_famd_path {X_famd_path}: {e}")
