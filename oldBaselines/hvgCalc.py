
import os
import numpy as np
import pandas as pd
import scanpy as sc

# ==========================================
# Configuration
# ==========================================

fname = "lungsCAN"


SPATIAL_PATH = f"/PASTE/YOUR/PATH/{fname}/spatial.h5ad"
    
OUTPUT_DIR = f"HVG_results/{fname}"

TOP_PERCENTAGES = [1, 5, 10, 25]

# Choose your expression source:
# "X"      -> adata.X
# "log1p"  -> adata.layers["log1p"]seq  
EXPRESSION_SOURCE = "X"

# True: data is already normalized and log1p
# False: normalize and log1p before HVG
ALREADY_LOG_NORMALIZED = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# Seurat HVG calculation
# ==========================================

def calculate_hvg_seurat(adata):

    print("Calculating Seurat-based HVGs...")

    # Calculate normalized dispersion scores
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=adata.n_vars,
        n_bins=20,
        subset=False,
        inplace=True
    )

    # Collect HVG scores
    hvg_df = pd.DataFrame({
        "Gene": adata.var_names.to_numpy(),
        "Mean": adata.var["means"].to_numpy(),
        "Dispersion": adata.var["dispersions"].to_numpy(),
        "Dispersion_norm":
            adata.var["dispersions_norm"].to_numpy()
    })

    # Remove genes without valid scores
    hvg_df = hvg_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["Dispersion_norm"])

    # Rank genes by normalized dispersion
    hvg_df = hvg_df.sort_values(
        by="Dispersion_norm",
        ascending=False
    ).reset_index(drop=True)

    hvg_df["Rank"] = np.arange(
        1, len(hvg_df) + 1
    )

    return hvg_df


# ==========================================
# Main
# ==========================================

def main():

    print("Loading spatial data...")

    adata = sc.read_h5ad(SPATIAL_PATH)

    adata.var_names = pd.Index(
        adata.var_names.astype(str)
    ).str.strip().str.upper()

    adata.var_names_make_unique()

    print("Original shape:", adata.shape)

    if EXPRESSION_SOURCE != "X":
        adata.X = adata.layers[
            EXPRESSION_SOURCE
        ].copy()

    # Prepare normalized log-expression
    if not ALREADY_LOG_NORMALIZED:
        sc.pp.normalize_total(
            adata,
            target_sum=1e4
        )
        sc.pp.log1p(adata)

    # Seurat HVG selection
    hvg_df = calculate_hvg_seurat(adata)

    # Save full ranking
    hvg_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "seurat_hvg_ranking.csv"
        ),
        index=False
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "ranked_genes.npy"
        ),
        hvg_df["Gene"].to_numpy(dtype=str)
    )

    total_genes = len(hvg_df)

    print("Genes with valid HVG scores:", total_genes)

    # Select top percentages
    for pct in TOP_PERCENTAGES:

        n_genes = max(
            1,
            int(np.ceil(
                total_genes * pct / 100
            ))
        )

        selected_genes = hvg_df[
            "Gene"
        ].iloc[:n_genes].to_numpy(dtype=str)

        save_path = os.path.join(
            OUTPUT_DIR,
            f"top_{pct}percent_hvg.npy"
        )

        np.save(save_path, selected_genes)

        print(
            f"Top {pct}%: {len(selected_genes)} genes"
        )

        print("Saved:", save_path)

    print("\nCompleted!")


if __name__ == "__main__":
    main()
