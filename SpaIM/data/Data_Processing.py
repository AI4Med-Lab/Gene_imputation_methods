# Import required libraries
import scanpy as sc  # Main library for single-cell RNA sequencing analysis
import numpy as np   # For numerical operations
import pandas as pd  # For data manipulation and analysis
import os            # For handling file paths and directories
from tqdm import tqdm  # For progress bar (not used in this snippet but imported)

def adata_to_cluster_expression(adata, scale=True):
    # Function to aggregate gene expression by cluster labels
    
    # Calculate frequency of each 'leiden' cluster label (normalized to proportions)
    value_counts = adata.obs['leiden'].value_counts(normalize=True)
    # Extract unique cluster labels from the index of the value counts
    unique_labels = value_counts.index
    # Create new observation metadata containing the unique cluster labels
    new_obs = pd.DataFrame({'leiden': unique_labels})
    # Print the unique cluster labels for verification
    print(unique_labels)
    # Create a new AnnData object with cluster-level observations, 
    # preserving original variables and unstructured metadata
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)
    # Initialize an empty array to store aggregated expression values
    # Dimensions: [number of clusters x number of genes]
    X_new = np.empty((len(unique_labels), adata.shape[1]))
    # Iterate through each unique cluster label
    for index, l, in enumerate(unique_labels):
        # Sum expression values across all cells in the current cluster (axis=0)
        X_new[index] = adata[adata.obs['leiden'] == l].X.sum(axis=0)
    # Assign the aggregated expression matrix to the new AnnData object
    adata_ret.X = X_new
    # Return the cluster-level AnnData object
    return adata_ret

# Define the root directory containing datasets
root = 'dataset/'
# Specify the name of the dataset to process
dataset_name = 'Dataset1'

# Read the single-cell RNA sequencing data (stored in h5ad format)
# from the specified dataset directory
scadata = sc.read(os.path.join(root, dataset_name, 'scRNA_count_cluster.h5ad'))
# Create a copy of the data to use for clustering analysis
adata_label = scadata.copy()
# Normalize total gene expression per cell (scales each cell to have the same total count)
sc.pp.normalize_total(adata_label)
# Apply log transformation to the normalized data (log(1 + x))
sc.pp.log1p(adata_label)
# Identify highly variable genes across cells (for dimensionality reduction)
sc.pp.highly_variable_genes(adata_label)
# Subset the data to keep only highly variable genes
adata_label = adata_label[:, adata_label.var.highly_variable]
# Scale gene expression values to have zero mean and unit variance,
# capping values at a maximum of 10 to reduce effect of outliers
sc.pp.scale(adata_label, max_value=10)
# Perform principal component analysis (PCA) for dimensionality reduction
sc.tl.pca(adata_label)
# Construct a neighborhood graph of cells based on PCA components
sc.pp.neighbors(adata_label)

# Iterate through different resolution parameters for Leiden clustering
# (higher resolution = more clusters)
for res in ['0.05', '0.10', '0.30', '0.50', '0.70', '0.90']:
    # Print current resolution value
    print(res)
    # Generate cluster labels using the Leiden algorithm with current resolution
    # random_state ensures reproducibility of results
    sc.tl.leiden(adata_label, resolution=float(res), random_state=1234)
    # Print the number of unique clusters generated at this resolution
    print(len(set(adata_label.obs['leiden'])))
    # Store the cluster labels in the original dataset with a resolution-specific column name
    scadata.obs['leiden_%s'%(res)] = adata_label.obs['leiden']

# Write the modified dataset (with all cluster labels) back to the original file
scadata.write(os.path.join(root, dataset_name, 'scRNA_count_cluster.h5ad'))
