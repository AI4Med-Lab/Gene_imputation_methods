"""Main module."""
import warnings
from telnetlib import X3PAD
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.nn import ModuleList

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, MultiDecoder, MultiEncoder, one_hot

import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import DataLoader
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from scipy.spatial import Delaunay
from . import faiss_neig

torch.backends.cudnn.benchmark = True


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_sparse
from torch import FloatTensor
import scanpy as sc


class H2GCN(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int = 1,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True
    ):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x: FloatTensor, adj: torch.sparse.Tensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return r_final


def corrcoef(x):
    """
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)
    return c

def extract_edge_index(
    adata: AnnData,
    batch_key: Optional[str] = None,
    spatial_key: Optional[str] = 'spatial',
    method: str = 'knn',
    n_neighbors: int = 15
    ):
    """
    Define edge_index for SIMVI model training.

    Args:
    ----
        adata: AnnData object.
        batch_key: Key in `adata.obs` for batch information. If batch_key is none,
        assume the adata is from the same batch. Otherwise, we create edge_index
        based on each batch and concatenate them.
        spatial_key: Key in `adata.obsm` for spatial location.
        method: method for establishing the graph proximity relationship between
        cells. Two available methods are: knn and Delouney. Knn is used as default
        due to its flexible neighbor number selection.
        n_neighbors: The number of n_neighbors of knn graph. Not used if the graph
        is based on Delouney triangularization.

    Returns
    -------
        edge_index: torch.Tensor.
    """
    if batch_key is not None:
        j = 0
        for i in adata.obs[batch_key].unique():
            adata_tmp = adata[adata.obs[batch_key]==i].copy()
            if method == 'knn':
                # A = kneighbors_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                A =  faiss_neig.knn_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                edge_index_tmp, edge_weight = from_scipy_sparse_matrix(A)
                label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                edge_index_tmp = label[edge_index_tmp]
                if j == 0:
                    edge_index = edge_index_tmp
                    j = 1
                else:
                    edge_index = torch.cat((edge_index,edge_index_tmp),1)

            else:
                tri = Delaunay(adata_tmp.obsm[spatial_key])
                triangles = tri.simplices
                edges = set()
                for triangle in triangles:
                    for i in range(3):
                        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                        edges.add(edge)
                edge_index_tmp = torch.tensor(list(edges)).t().contiguous()
                label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                edge_index_tmp = label[edge_index_tmp]
                if j == 0:
                    edge_index = edge_index_tmp
                    j = 1
                else:
                    edge_index = torch.cat((edge_index,edge_index_tmp),1)
    else:
        if method == 'knn':
            # print(adata)
            # print(adata.obsm)
            # A = kneighbors_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
            A =  faiss_neig.knn_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
            edge_index, edge_weight = from_scipy_sparse_matrix(A)
        else:
            tri = Delaunay(adata.obsm[spatial_key])
            triangles = tri.simplices
            edges = set()
            for triangle in triangles:
                for i in range(3):
                    edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                    edges.add(edge)
            edge_index = torch.tensor(list(edges)).t().contiguous()

    return edge_index

class JVAE(BaseModuleClass):
    """Joint variational auto-encoder for imputing missing genes in spatial data.

    Implementation of gimVI :cite:p:`Lopez19`.

    Parameters
    ----------
    dim_input_list
        List of number of input genes for each dataset. If
            the datasets have different sizes, the dataloader will loop on the
            smallest until it reaches the size of the longest one
    total_genes
        Total number of different genes
    indices_mappings
        list of mapping the model inputs to the model output
        Eg: ``[[0,2], [0,1,3,2]]`` means the first dataset has 2 genes that will be reconstructed at location ``[0,2]``
        the second dataset has 4 genes that will be reconstructed at ``[0,1,3,2]``
    gene_likelihoods
        list of distributions to use in the generative process 'zinb', 'nb', 'poisson'
    model_library_bools bool list
        model or not library size with a latent variable or use observed values
    library_log_means np.ndarray list
        List of 1 x n_batch array of means of the log library sizes.
        Parameterizes prior on library size if not using observed library sizes.
    library_log_vars np.ndarray list
        List of 1 x n_batch array of variances of the log library sizes.
        Parameterizes prior on library size if not using observed library sizes.
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    n_layers_encoder_shared
        number of shared layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    n_layers_decoder_individual
        number of layers that are conditionally batchnormed in the encoder
    n_layers_decoder_shared
        number of shared layers in the decoder
    dim_hidden_decoder_individual
        dimension of the individual hidden layers in the decoder
    dim_hidden_decoder_shared
        dimension of the shared hidden layers in the decoder
    dropout_rate_encoder
        dropout encoder
    dropout_rate_decoder
        dropout decoder
    n_batch
        total number of batches
    n_labels
        total number of labels
    dispersion
        See ``vae.py``
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        indices_mappings: List[Union[np.ndarray, slice]],
        gene_likelihoods: List[str],
        model_library_bools: List[bool],
        library_log_means: List[Optional[np.ndarray]],
        library_log_vars: List[Optional[np.ndarray]],
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 1,
        dim_hidden_encoder: int = 64,
        n_layers_decoder_individual: int = 0,
        n_layers_decoder_shared: int = 0,
        dim_hidden_decoder_individual: int = 64,
        dim_hidden_decoder_shared: int = 64,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float = 0.2,
        n_batch: int = 0,
        n_labels: int = 0,
        dispersion: str = "gene-batch",
        log_variational: bool = True,
        correlation_const = False,
        spatial_data = None,
        neighbor_size = None,
        rna_data = None,
        graph_residual_strength: float = 0.05,
        graph_scale_min: float = 1e-4,
        graph_scale_max: float = 10.0,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.correlation_const = correlation_const
        self.total_genes = total_genes
        self.indices_mappings = indices_mappings
        self.gene_likelihoods = gene_likelihoods
        self.model_library_bools = model_library_bools
        self.neighbor_size = neighbor_size
        self.rna_data = rna_data

        # Numerical-stability controls for the spatial graph posterior update.
        self.graph_residual_strength = float(graph_residual_strength)
        self.graph_scale_min = float(graph_scale_min)
        self.graph_scale_max = float(graph_scale_max)
        if not (0.0 < self.graph_scale_min < self.graph_scale_max):
            raise ValueError(
                "Require 0 < graph_scale_min < graph_scale_max, got "
                f"{self.graph_scale_min} and {self.graph_scale_max}."
            )

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for mode in range(len(dim_input_list)):
            if self.model_library_bools[mode]:
                self.register_buffer(
                    f"library_log_means_{mode}",
                    torch.from_numpy(library_log_means[mode]).float(),
                )
                self.register_buffer(
                    f"library_log_vars_{mode}",
                    torch.from_numpy(library_log_vars[mode]).float(),
                )

        self.n_latent = n_latent
        self.spatial_data = spatial_data

        self.n_batch = n_batch
        self.n_labels = n_labels

        self.dispersion = dispersion
        self.log_variational = log_variational

        self.z_encoder = MultiEncoder(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
            return_dist=True,
        )

        self.graph_encoder_loc = GATConv(self.n_latent, self.n_latent)
        self.graph_encoder_scale = GATConv(self.n_latent, self.n_latent)
        self.graph_encoder_z = GATConv(self.n_latent, self.n_latent)


        self.l_encoders = ModuleList(
            [
                Encoder(
                    self.n_input_list[i],
                    1,
                    n_layers=1,
                    dropout_rate=dropout_rate_encoder,
                    return_dist=True,
                )
                if self.model_library_bools[i]
                else None
                for i in range(len(self.n_input_list))
            ]
        )

        self.decoder = MultiDecoder(
            self.n_latent, 
            self.total_genes,
            n_hidden_conditioned=dim_hidden_decoder_individual,
            n_hidden_shared=dim_hidden_decoder_shared,
            n_layers_conditioned=n_layers_decoder_individual,
            n_layers_shared=n_layers_decoder_shared,
            n_cat_list=[self.n_batch],
            dropout_rate=dropout_rate_decoder,
        )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_labels))
        else:  # gene-cell
            pass


    def _refine_spatial_posterior(
        self,
        qz: Normal,
        indices: Optional[torch.Tensor],
    ) -> Tuple[Normal, torch.Tensor]:
        """
        Refine the spatial posterior with a bounded GAT residual.

        The original VISTA graph update acted on qz.scale with an unrestricted
        exponential. That can overflow. Here the scale is updated in log-space,
        with a bounded residual, and the graph-refined posterior is used to draw
        the latent z that is passed to the decoder.
        """
        if indices is None or self.spatial_data is None:
            return qz, qz.rsample()

        idx = (
            indices.detach()
            .view(-1)
            .to("cpu")
            .numpy()
            .astype(np.int64, copy=False)
        )
        if idx.size < 2:
            return qz, qz.rsample()

        # Guard against invalid fold-local row indices.
        valid = (idx >= 0) & (idx < self.spatial_data.n_obs)
        if not np.all(valid):
            warnings.warn(
                "Some spatial row indices were outside the registered "
                "spatial AnnData. Invalid rows are ignored for graph refinement."
            )
            idx = idx[valid]

        if idx.size < 2:
            return qz, qz.rsample()

        adata_new = self.spatial_data[idx].copy()
        n_neighbors = min(
            int(self.neighbor_size),
            max(int(idx.size) - 1, 0),
        )

        if n_neighbors < 1:
            return qz, qz.rsample()

        edge_index = extract_edge_index(
            adata_new,
            spatial_key="spatial",
            n_neighbors=n_neighbors,
        ).to(qz.loc.device)

        # Defensive edge filtering for FAISS / graph edge cases.
        if edge_index.numel() == 0:
            return qz, qz.rsample()

        edge_index = edge_index.long()
        n_nodes = qz.loc.shape[0]
        edge_valid = (
            (edge_index[0] >= 0)
            & (edge_index[0] < n_nodes)
            & (edge_index[1] >= 0)
            & (edge_index[1] < n_nodes)
        )
        edge_index = edge_index[:, edge_valid]

        if edge_index.numel() == 0:
            return qz, qz.rsample()

        # Mean update: bounded additive residual.
        loc_raw = self.graph_encoder_loc(qz.loc, edge_index)
        loc_delta = self.graph_residual_strength * torch.tanh(loc_raw)
        new_loc = qz.loc + loc_delta

        # Scale update: operate in log-scale, then exponentiate only after
        # clamping the log-scale to a safe range.
        base_scale = torch.clamp(
            qz.scale,
            min=self.graph_scale_min,
            max=self.graph_scale_max,
        )
        log_scale = torch.log(base_scale)

        scale_raw = self.graph_encoder_scale(log_scale, edge_index)
        scale_delta = self.graph_residual_strength * torch.tanh(scale_raw)
        new_log_scale = log_scale + scale_delta

        log_min = float(np.log(self.graph_scale_min))
        log_max = float(np.log(self.graph_scale_max))
        new_log_scale = torch.nan_to_num(
            new_log_scale,
            nan=0.0,
            posinf=log_max,
            neginf=log_min,
        )
        new_log_scale = torch.clamp(
            new_log_scale,
            min=log_min,
            max=log_max,
        )

        new_loc = torch.nan_to_num(
            new_loc,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        new_loc = torch.clamp(new_loc, min=-20.0, max=20.0)

        new_scale = torch.exp(new_log_scale)
        new_scale = torch.clamp(
            new_scale,
            min=self.graph_scale_min,
            max=self.graph_scale_max,
        )

        qz_graph = Normal(new_loc, new_scale)
        z_graph = qz_graph.rsample()
        return qz_graph, z_graph

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        mode: int = None,
        deterministic: bool = False,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample tensor of latent values from the posterior."""
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode when having multiple datasets")

        outputs = self.inference(x, mode, indices=indices)
        qz_m = outputs["qz"].loc
        z = outputs["z"]
        if deterministic:
            z = qz_m
        return z

    def sample_from_posterior_l(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``
        """
        inference_out = self.inference(x, mode)
        return (
            inference_out["ql"].loc
            if (deterministic and inference_out["ql"] is not None)
            else inference_out["library"]
        )
    
    def sample_theta(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: int = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the tensor of dispersion parameters."""
        gen_out = self._run_forward(
            x,
            mode,
            batch_index,
            y=y,
            deterministic=deterministic,
            decode_mode=decode_mode,
            indices=indices,
        )
        return gen_out["px_r"]

    def sample_scale(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return predicted normalized expression frequencies."""
        gen_out = self._run_forward(
            x,
            mode,
            batch_index,
            y=y,
            deterministic=deterministic,
            decode_mode=decode_mode,
            indices=indices,
        )
        return gen_out["px_scale"]

    # This is a potential wrapper for a vae like get_sample_rate
    def get_sample_rate(self, x, batch_index, *_, **__):
        """Get the sample rate for the model."""
        return self.sample_rate(x, 0, batch_index)

    def _run_forward(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: int = None,
        indices: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run inference followed by the decoder."""
        if decode_mode is None:
            decode_mode = mode

        inference_out = self.inference(
            x,
            mode,
            indices=indices,
        )

        if deterministic:
            z = inference_out["qz"].loc
            if inference_out["ql"] is not None:
                library = inference_out["ql"].loc
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            library = inference_out["library"]

        return self.generative(
            z,
            library,
            batch_index,
            y,
            decode_mode,
        )

    def sample_rate(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: int = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return predicted expression rates."""
        gen_out = self._run_forward(
            x,
            mode,
            batch_index,
            y=y,
            deterministic=deterministic,
            decode_mode=decode_mode,
            indices=indices,
        )
        return gen_out["px_rate"]

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        """Compute the reconstruction loss."""
        reconstruction_loss = None
        if self.gene_likelihoods[mode] == "zinb":
            reconstruction_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "nb":
            reconstruction_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "poisson":
            reconstruction_loss = -Poisson(px_rate).log_prob(x).sum(dim=1)
        return reconstruction_loss

    def _get_inference_input(self, tensors):
        """Get inputs for the inference model, including fold-local row indices."""
        return {
            "x": tensors[REGISTRY_KEYS.X_KEY],
            "indices": tensors.get(REGISTRY_KEYS.INDICES_KEY, None),
        }

    def _get_generative_input(self, tensors, inference_outputs):
        """Get inputs for the generative model."""
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        return {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "y": y,
        }

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        mode: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run the inference model with a stable spatial graph refinement."""
        x_ = x
        if self.log_variational:
            # VISTA expects count-like non-negative input.
            x_ = torch.log1p(torch.clamp_min(x_, 0.0))

        qz, z = self.z_encoder(x_, mode)

        # IMPORTANT:
        # Apply graph refinement here (not only inside loss) so the graph-refined
        # latent is also used by the decoder and by deterministic imputation.
        if mode == 1:
            qz, z = self._refine_spatial_posterior(qz, indices)

        ql, library = None, None
        if self.model_library_bools[mode]:
            ql, library = self.l_encoders[mode](x_)
        else:
            library_sum = torch.sum(torch.clamp_min(x, 0.0), dim=1)
            library_sum = torch.clamp(library_sum, min=1e-8)
            library = torch.log(library_sum).view(-1, 1)

        return {
            "qz": qz,
            "z": z,
            "ql": ql,
            "library": library,
        }

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> dict:
        """Run the generative model."""
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            z, mode, library, self.dispersion, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r.view(1, self.px_r.size(0))
        # Keep dispersion positive while avoiding exp overflow.
        px_r = torch.exp(torch.clamp(px_r, min=-20.0, max=20.0))
        px_r = torch.nan_to_num(
            px_r,
            nan=1.0,
            posinf=float(np.exp(20.0)),
            neginf=float(np.exp(-20.0)),
        )

        # Stable normalization over the genes observed in this mode.
        denom = torch.sum(
            px_scale[:, self.indices_mappings[mode]],
            dim=1,
        ).view(-1, 1)
        denom = torch.clamp(denom, min=1e-8)
        px_scale = px_scale / denom
        px_scale = torch.nan_to_num(
            px_scale,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        # exp(library) is the library-size multiplier.
        safe_library = torch.clamp(library, min=-20.0, max=20.0)
        px_rate = px_scale * torch.exp(safe_library)
        px_rate = torch.nan_to_num(
            px_rate,
            nan=1e-8,
            posinf=float(np.exp(20.0)),
            neginf=1e-8,
        )
        px_rate = torch.clamp(px_rate, min=1e-8)

        return {
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }
    
    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        mode: Optional[int] = None,
        kl_weight=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network


        Returns
        -------
        the reconstruction loss and the Kullback divergences
        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode")
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        qz = inference_outputs["qz"]
        # For spatial mode, qz has already been graph-refined in inference().
        # This ensures the same posterior is used for reconstruction, KL, the
        # adversarial latent, and deterministic inference.
        ql = inference_outputs["ql"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]

        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )

        # KL Divergence
        mean = torch.zeros_like(qz.loc)
        scale = torch.ones_like(qz.scale)
        kl_divergence_z = kl(qz, Normal(mean, scale)).sum(dim=1)

        if self.model_library_bools[mode]:
            library_log_means = getattr(self, f"library_log_means_{mode}")
            library_log_vars = getattr(self, f"library_log_vars_{mode}")

            local_library_log_means = F.linear(
                one_hot(batch_index, self.n_batch), library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, self.n_batch), library_log_vars
            )
            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local = kl_divergence_l + kl_divergence_z
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)
        return LossOutput(
            loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_local
        )