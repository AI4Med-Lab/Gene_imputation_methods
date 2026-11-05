"""Modular PASTA gene-holdout benchmarking implementation."""
from __future__ import annotations

import gc
import logging
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from spatial import prepare_neighbor_stats

LOGGER = logging.getLogger("PASTA-baseline")


class PastaMapper:
    """
    Low-GPU-memory PASTA mapper.

    The logical mapping is still:
        M.shape = [n_scRNA, n_spatial]

    but M is stored as row chunks on CPU. During training only ONE chunk
    is copied to the selected GPU at a time. Adam first/second-moment
    states are also kept on CPU.

    This allows the full scRNA reference to be used without placing the
    complete 31k x 139k mapping and Adam states in GPU memory.
    """

    def __init__(
        self,
        n_sc: int,
        n_sp: int,
        device: torch.device,
        seed: int,
        m_chunk_rows: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_sc = int(n_sc)
        self.n_sp = int(n_sp)
        self.device = torch.device(device)
        self.m_chunk_rows = int(m_chunk_rows)
        self.dtype = dtype

        if self.m_chunk_rows <= 0:
            raise ValueError("m_chunk_rows must be > 0")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        self.M_chunks: List[torch.Tensor] = []
        self.chunk_ranges: List[Tuple[int, int]] = []

        for start in range(0, self.n_sc, self.m_chunk_rows):
            end = min(start + self.m_chunk_rows, self.n_sc)

            m = torch.empty(
                (end - start, self.n_sp),
                dtype=self.dtype,
                device="cpu",
            )
            m.normal_(mean=0.0, std=1.0, generator=gen)

            self.M_chunks.append(m)
            self.chunk_ranges.append((start, end))

        # Adam states are created lazily on CPU during the first update.
        self.exp_avg: List[Optional[torch.Tensor]] = [
            None for _ in self.M_chunks
        ]
        self.exp_avg_sq: List[Optional[torch.Tensor]] = [
            None for _ in self.M_chunks
        ]
        self.adam_step = 0

        matrix_gib = (
            self.n_sc * self.n_sp * 4 / (1024 ** 3)
        )

        LOGGER.info(
            "CPU-backed PASTA M-chunk enabled | "
            "%d chunks | chunk_rows=%d | logical M=%d x %d",
            len(self.M_chunks),
            self.m_chunk_rows,
            self.n_sc,
            self.n_sp,
        )
        LOGGER.info(
            "M storage ~= %.2f GiB CPU; Adam states add ~= %.2f GiB "
            "(~%.2f GiB CPU total after first optimizer step).",
            matrix_gib,
            2.0 * matrix_gib,
            3.0 * matrix_gib,
        )

    @staticmethod
    def _contribution(
        m_gpu: torch.Tensor,
        s_gpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        One exact row-block contribution to:
            softmax(M, dim=1).T @ S
        """
        prob = torch.softmax(m_gpu, dim=1)
        return prob.transpose(0, 1).matmul(s_gpu)

    @torch.no_grad()
    def project_no_grad(
        self,
        S_gpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the complete PASTA projection without building a graph
        through M. Only one M chunk is present on GPU at a time.
        """
        out = torch.zeros(
            (self.n_sp, S_gpu.shape[1]),
            device=self.device,
            dtype=S_gpu.dtype,
        )

        for m_cpu, (start, end) in zip(
            self.M_chunks,
            self.chunk_ranges,
        ):
            m_gpu = m_cpu.to(
                self.device,
                dtype=torch.float32,
                non_blocking=False,
            )

            contrib = self._contribution(
                m_gpu,
                S_gpu[start:end],
            )

            out.add_(contrib)

            del m_gpu, contrib

        return out

    @torch.no_grad()
    def project_test(
        self,
        S_test_gpu: torch.Tensor,
    ) -> torch.Tensor:
        return self.project_no_grad(S_test_gpu)

    def adam_update_chunk(
        self,
        chunk_id: int,
        grad_cpu: torch.Tensor,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """
        Exact standard Adam update for one CPU-resident M chunk.
        """
        m = self.M_chunks[chunk_id]

        if self.exp_avg[chunk_id] is None:
            self.exp_avg[chunk_id] = torch.zeros_like(
                m,
                device="cpu",
            )
            self.exp_avg_sq[chunk_id] = torch.zeros_like(
                m,
                device="cpu",
            )

        exp_avg = self.exp_avg[chunk_id]
        exp_avg_sq = self.exp_avg_sq[chunk_id]

        exp_avg.mul_(beta1).add_(
            grad_cpu,
            alpha=1.0 - beta1,
        )

        exp_avg_sq.mul_(beta2).addcmul_(
            grad_cpu,
            grad_cpu,
            value=1.0 - beta2,
        )

        bias_correction1 = 1.0 - beta1 ** self.adam_step
        bias_correction2 = 1.0 - beta2 ** self.adam_step

        step_size = lr / bias_correction1

        # Temporary is only one M chunk, not the full mapping.
        denom = (
            exp_avg_sq.sqrt()
            / math.sqrt(bias_correction2)
        ).add_(eps)

        m.addcdiv_(
            exp_avg,
            denom,
            value=-step_size,
        )

        del denom


def pasta_loss(
    G_pred: torch.Tensor,
    G_true: torch.Tensor,
    pathway_mask: torch.Tensor,
    neighbor_stats,
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    lambda_4: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # L_global: maximize per-gene cosine similarity across ST cells.
    global_sim = F.cosine_similarity(
        G_pred,
        G_true,
        dim=0,
        eps=1e-8,
    ).mean()

    pred_path = G_pred[:, pathway_mask]
    true_path = G_true[:, pathway_mask]

    pathway_sim = F.cosine_similarity(
        pred_path,
        true_path,
        dim=0,
        eps=1e-8,
    ).mean()

    mse_terms = []
    neighbor_cos_terms = []

    for idx, nbr_mean, nbr_mean_sq, nbr_unit_mean in neighbor_stats:
        pred_ct = pred_path[idx]

        # E[(N - pred)^2]
        mse = (
            pred_ct.square()
            - 2.0 * pred_ct * nbr_mean
            + nbr_mean_sq
        ).mean()
        mse_terms.append(mse)

        pred_unit = F.normalize(
            pred_ct,
            p=2,
            dim=1,
            eps=1e-8,
        )

        neighbor_cos_terms.append(
            (
                pred_unit
                * nbr_unit_mean
            ).sum(dim=1).mean()
        )

    if mse_terms:
        recon = torch.stack(mse_terms).mean()
        neighbor_sim = torch.stack(
            neighbor_cos_terms
        ).mean()
    else:
        recon = torch.zeros(
            (),
            device=G_pred.device,
            dtype=G_pred.dtype,
        )
        neighbor_sim = torch.zeros(
            (),
            device=G_pred.device,
            dtype=G_pred.dtype,
        )

    total = (
        lambda_1 * recon
        - lambda_2 * global_sim
        - lambda_3 * pathway_sim
        - lambda_4 * neighbor_sim
    )

    parts = {
        "loss": float(total.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "global_sim": float(global_sim.detach().cpu()),
        "pathway_sim": float(pathway_sim.detach().cpu()),
        "neighbor_sim": float(neighbor_sim.detach().cpu()),
    }

    return total, parts


def train_pasta(
    S_train_np: np.ndarray,
    G_train_np: np.ndarray,
    pathway_mask_np: np.ndarray,
    neighbor_groups,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    lambdas: Tuple[float, float, float, float],
    print_every: int,
    m_chunk_rows: int,
    optimizer_name: str,
) -> Tuple[PastaMapper, List[Dict[str, float]]]:

    if optimizer_name != "adam":
        raise ValueError(
            "The CPU-backed full-reference M-chunk implementation "
            "currently supports --optimizer adam only."
        )

    # S and G are small compared with M and can remain on GPU.
    S = torch.as_tensor(
        S_train_np,
        device=device,
        dtype=torch.float32,
    )

    G = torch.as_tensor(
        G_train_np,
        device=device,
        dtype=torch.float32,
    )

    pathway_mask = torch.as_tensor(
        pathway_mask_np,
        device=device,
        dtype=torch.bool,
    )

    if int(pathway_mask.sum().item()) == 0:
        raise ValueError(
            "No visible/training genes overlap the selected pathway."
        )

    model = PastaMapper(
        n_sc=S.shape[0],
        n_sp=G.shape[0],
        device=device,
        seed=seed,
        m_chunk_rows=m_chunk_rows,
    )

    G_pathway = G[:, pathway_mask]

    neighbor_stats = prepare_neighbor_stats(
        G_pathway,
        neighbor_groups,
        device,
    )

    history: List[Dict[str, float]] = []

    lambda_1, lambda_2, lambda_3, lambda_4 = lambdas

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ==============================================================
        # Stage A
        # Compute the exact complete G_pred with all M chunks, but do not
        # retain an M->G_pred graph.
        # ==============================================================
        with torch.no_grad():
            G_pred_value = model.project_no_grad(S)

        # ==============================================================
        # Stage B
        # Differentiate the original PASTA objective only with respect
        # to G_pred. This graph is small: [n_spatial x n_train_genes].
        # ==============================================================
        G_pred_leaf = (
            G_pred_value
            .detach()
            .requires_grad_(True)
        )

        loss, parts = pasta_loss(
            G_pred_leaf,
            G,
            pathway_mask,
            neighbor_stats,
            lambda_1,
            lambda_2,
            lambda_3,
            lambda_4,
        )

        grad_G_pred = torch.autograd.grad(
            outputs=loss,
            inputs=G_pred_leaf,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

        del loss
        del G_pred_leaf
        del G_pred_value

        # Every M chunk gets exactly one Adam update per epoch.
        model.adam_step += 1

        # ==============================================================
        # Stage C
        # Recompute ONE M chunk on GPU, propagate dL/dG_pred through
        # only that chunk, transfer its gradient to CPU, update the
        # CPU-resident mapping chunk, then release GPU temporaries.
        # ==============================================================
        for chunk_id, (m_cpu, (start, end)) in enumerate(
            zip(model.M_chunks, model.chunk_ranges)
        ):
            m_gpu = (
                m_cpu
                .to(
                    device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
                .detach()
                .requires_grad_(True)
            )

            contribution = model._contribution(
                m_gpu,
                S[start:end],
            )

            grad_m_gpu = torch.autograd.grad(
                outputs=contribution,
                inputs=m_gpu,
                grad_outputs=grad_G_pred,
                retain_graph=False,
                create_graph=False,
            )[0]

            grad_cpu = (
                grad_m_gpu
                .detach()
                .cpu()
            )

            model.adam_update_chunk(
                chunk_id=chunk_id,
                grad_cpu=grad_cpu,
                lr=lr,
            )

            del (
                m_gpu,
                contribution,
                grad_m_gpu,
                grad_cpu,
            )

        del grad_G_pred

        parts["epoch"] = epoch
        parts["seconds"] = time.time() - t0
        history.append(parts)

        if (
            epoch == 1
            or epoch % print_every == 0
            or epoch == epochs
        ):
            LOGGER.info(
                "Epoch %04d | loss=%.6f | recon=%.6f | "
                "global=%.4f | pathway=%.4f | neighbor=%.4f | %.1fs",
                epoch,
                parts["loss"],
                parts["recon"],
                parts["global_sim"],
                parts["pathway_sim"],
                parts["neighbor_sim"],
                parts["seconds"],
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    del S, G, G_pathway, neighbor_stats
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model, history


def project_test_genes(
    model: PastaMapper,
    S_test_np: np.ndarray,
    device: torch.device,
    m_chunk_rows: int,
) -> np.ndarray:

    S_test = torch.as_tensor(
        S_test_np,
        device=device,
        dtype=torch.float32,
    )

    pred = model.project_test(S_test)

    out = (
        pred
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    del S_test, pred

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out


def mapping_memory_report(n_sc: int, n_sp: int, optimizer_name: str, device: torch.device) -> Dict[str, float]:
    entries = int(n_sc) * int(n_sp)
    matrix_gib = entries * 4 / (1024 ** 3)
    # Persistent-ish tensors: M + grad + Adam m + Adam v. Chunked checkpointing
    # prevents a second full mapping-probability matrix from being retained.
    multiplier = 4 if optimizer_name == "adam" else 2
    persistent_gib = matrix_gib * multiplier
    report = {
        "mapping_entries": entries,
        "mapping_matrix_gib": matrix_gib,
        "rough_persistent_training_gib": persistent_gib,
    }
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        report["gpu_total_gib"] = float(total)
    return report
