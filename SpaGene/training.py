"""SpaGene autoencoder pretraining, translation training, and inference."""
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from utils import make_loader, safe_corr

def amp_context(device, enabled):
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        )
    return nullcontext()

def pearson_loss(pred, target, eps=1e-8):
    """1 - mean feature-wise Pearson correlation."""
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    num = (pred * target).sum(dim=0)
    den = torch.sqrt((pred.square().sum(dim=0) + eps) * (target.square().sum(dim=0) + eps))
    corr = num / den
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - corr.mean()


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def hinge_discriminator_loss(real_score, fake_score):
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def gradient_penalty(discriminator, real, fake):
    """Standard interpolation gradient penalty in latent space."""
    alpha_shape = [real.shape[0]] + [1] * (real.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real.device, dtype=real.dtype)
    inter = alpha * real + (1.0 - alpha) * fake
    inter.requires_grad_(True)
    score = discriminator(inter)
    grad = torch.autograd.grad(
        outputs=score.sum(),
        inputs=inter,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(grad.shape[0], -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


@torch.no_grad()
def reconstruction_mse(encoder, decoder, x, batch_size, device):
    encoder.eval()
    decoder.eval()
    loader = make_loader(x, batch_size, shuffle=False, drop_last=False)
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pred = decoder(encoder(batch))
        total += F.mse_loss(pred, batch, reduction="sum").item()
        n += batch.numel()
    return total / max(n, 1)


def train_autoencoder(encoder, decoder, x, epochs, batch_size, lr, device, use_amp, label):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))
    loader = make_loader(x, batch_size, shuffle=True, drop_last=True)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        running = 0.0
        batches = 0

        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                pred = decoder(encoder(batch))
                loss = F.mse_loss(pred, batch)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach().cpu())
            batches += 1

        eval_loss = reconstruction_mse(encoder, decoder, x, batch_size, device)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  [{label} AE] epoch {epoch:03d}/{epochs} "
                f"train={running/max(batches,1):.6f} recon={eval_loss:.6f}"
            )

    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    return float(best_loss)


@torch.no_grad()
def predict_full(model, x_st, batch_size, device):
    model.eval()
    loader = make_loader(x_st, batch_size, shuffle=False, drop_last=False)
    out = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pred = model.predict_st_to_sc(batch)
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def visible_gene_corr(model, x_st, train_indices_in_sc, batch_size, device):
    pred = predict_full(model, x_st, batch_size, device)
    pred_visible = pred[:, train_indices_in_sc]
    vals = [
        safe_corr(pearsonr, x_st[:, j], pred_visible[:, j])
        for j in range(x_st.shape[1])
    ]
    return float(np.nanmean(vals))


def train_translation(
    model,
    x_st,
    x_sc,
    train_indices_in_sc,
    args,
    device,
):
    # SpaGene freezes the domain autoencoders before translation training.
    for module in [model.enc_st, model.dec_st, model.enc_sc, model.dec_sc]:
        module.eval()
        set_requires_grad(module, False)

    trans_params = list(model.trans_st2sc.parameters()) + list(model.trans_sc2st.parameters())
    disc_params = list(model.disc_st.parameters()) + list(model.disc_sc.parameters())

    opt_t = torch.optim.Adam(trans_params, lr=args.lr_trans, weight_decay=args.weight_decay)
    opt_d = torch.optim.Adam(disc_params, lr=args.lr_disc, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt_t, step_size=10, gamma=0.5)

    scaler_t = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    scaler_d = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    st_loader = make_loader(x_st, args.batch_size, shuffle=True, drop_last=True)
    sc_loader = make_loader(x_sc, args.batch_size, shuffle=True, drop_last=True)

    best_corr = -np.inf
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.trans_st2sc.train()
        model.trans_sc2st.train()
        model.disc_st.train()
        model.disc_sc.train()

        st_iter = iter(st_loader)
        sc_iter = iter(sc_loader)
        n_steps = max(len(st_loader), len(sc_loader))

        sum_t = 0.0
        sum_d = 0.0

        for _ in range(n_steps):
            try:
                xs = next(st_iter)
            except StopIteration:
                st_iter = iter(st_loader)
                xs = next(st_iter)
            try:
                xt = next(sc_iter)
            except StopIteration:
                sc_iter = iter(sc_loader)
                xt = next(sc_iter)

            # Paired batch size is not required by SpaGene. GP interpolation does
            # require equal sizes, so use the common prefix only for GP.
            xs = xs.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            with torch.no_grad():
                z_st = model.enc_st(xs)
                z_sc = model.enc_sc(xt)

            # -------------------------------------------------------------
            # Discriminator update
            # -------------------------------------------------------------
            set_requires_grad(model.disc_st, True)
            set_requires_grad(model.disc_sc, True)
            opt_d.zero_grad(set_to_none=True)

            with amp_context(device, args.amp):
                fake_sc = model.trans_st2sc(z_st).detach()
                fake_st = model.trans_sc2st(z_sc).detach()

                d_sc = hinge_discriminator_loss(model.disc_sc(z_sc), model.disc_sc(fake_sc))
                d_st = hinge_discriminator_loss(model.disc_st(z_st), model.disc_st(fake_st))
                d_loss = args.lambda_disc * (d_sc + d_st)

            # Gradient penalty in fp32 for numerical stability.
            if args.lambda_gp > 0:
                n_gp_sc = min(z_sc.shape[0], fake_sc.shape[0])
                n_gp_st = min(z_st.shape[0], fake_st.shape[0])
                gp_sc = gradient_penalty(
                    model.disc_sc,
                    z_sc[:n_gp_sc].float(),
                    fake_sc[:n_gp_sc].float(),
                )
                gp_st = gradient_penalty(
                    model.disc_st,
                    z_st[:n_gp_st].float(),
                    fake_st[:n_gp_st].float(),
                )
                d_loss = d_loss + args.lambda_gp * (gp_sc + gp_st)

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc_params, args.gradient_clip)
            scaler_d.step(opt_d)
            scaler_d.update()

            # -------------------------------------------------------------
            # Translator update
            # -------------------------------------------------------------
            set_requires_grad(model.disc_st, False)
            set_requires_grad(model.disc_sc, False)
            opt_t.zero_grad(set_to_none=True)

            with amp_context(device, args.amp):
                z_st2sc = model.trans_st2sc(z_st)
                z_sc2st = model.trans_sc2st(z_sc)

                # Adversarial generator loss.
                loss_adv = -model.disc_sc(z_st2sc).mean() - model.disc_st(z_sc2st).mean()

                # Identity/shared-gene anchoring.
                pred_st2sc = model.dec_sc(z_st2sc)
                pred_sc2st = model.dec_st(z_sc2st)
                loss_id_st = pearson_loss(pred_st2sc[:, train_indices_in_sc], xs)
                loss_id_sc = pearson_loss(pred_sc2st, xt[:, train_indices_in_sc])
                loss_id = loss_id_st + loss_id_sc

                # Cycle consistency in expression space.
                cyc_st = model.dec_st(model.trans_sc2st(z_st2sc))
                cyc_sc = model.dec_sc(model.trans_st2sc(z_sc2st))
                loss_cyc = F.mse_loss(cyc_st, xs) + F.mse_loss(cyc_sc, xt)

                t_loss = (
                    args.lambda_cyc * loss_cyc
                    + args.lambda_id * loss_id
                    + args.lambda_adv * loss_adv
                )

            scaler_t.scale(t_loss).backward()
            scaler_t.unscale_(opt_t)
            torch.nn.utils.clip_grad_norm_(trans_params, args.gradient_clip)
            scaler_t.step(opt_t)
            scaler_t.update()

            sum_t += float(t_loss.detach().cpu())
            sum_d += float(d_loss.detach().cpu())

        scheduler.step()

        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if do_eval:
            corr = visible_gene_corr(
                model, x_st, train_indices_in_sc, args.eval_batch_size, device
            )
            print(
                f"  [Translator] epoch {epoch:03d}/{args.epochs} "
                f"T={sum_t/max(n_steps,1):.6f} D={sum_d/max(n_steps,1):.6f} "
                f"visible_Pearson={corr:.5f}"
            )

            if np.isfinite(corr) and corr > best_corr + args.early_stop_min_delta:
                best_corr = corr
                patience_count = 0
                best_state = {
                    "trans_st2sc": {
                        k: v.detach().cpu().clone()
                        for k, v in model.trans_st2sc.state_dict().items()
                    },
                    "trans_sc2st": {
                        k: v.detach().cpu().clone()
                        for k, v in model.trans_sc2st.state_dict().items()
                    },
                    "disc_st": {
                        k: v.detach().cpu().clone()
                        for k, v in model.disc_st.state_dict().items()
                    },
                    "disc_sc": {
                        k: v.detach().cpu().clone()
                        for k, v in model.disc_sc.state_dict().items()
                    },
                }
            else:
                patience_count += 1

            if (
                args.early_stop_patience > 0
                and patience_count >= args.early_stop_patience
            ):
                print(
                    f"  Early stopping translator stage at epoch {epoch}; "
                    f"best visible Pearson={best_corr:.5f}"
                )
                break

    if best_state is not None:
        model.trans_st2sc.load_state_dict(best_state["trans_st2sc"])
        model.trans_sc2st.load_state_dict(best_state["trans_sc2st"])
        model.disc_st.load_state_dict(best_state["disc_st"])
        model.disc_sc.load_state_dict(best_state["disc_sc"])

    return float(best_corr)
