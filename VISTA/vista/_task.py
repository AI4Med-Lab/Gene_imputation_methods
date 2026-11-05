import torch

from scvi import REGISTRY_KEYS
from scvi.module import Classifier
from scvi.train import AdversarialTrainingPlan


def _assert_finite(name, value):
    """Fail early with a useful error instead of propagating NaN/Inf."""
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        finite = value[torch.isfinite(value)]
        if finite.numel() > 0:
            finite_min = float(finite.min().detach().cpu())
            finite_max = float(finite.max().detach().cpu())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
        raise FloatingPointError(
            f"{name} contains NaN/Inf "
            f"(finite min={finite_min}, finite max={finite_max})."
        )


class GIMVITrainingPlan(AdversarialTrainingPlan):
    """gimVI/VISTA training plan with early numerical checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs["adversarial_classifier"] is True:
            self.n_output_classifier = 2
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=3,
                logits=True,
            )
        else:
            self.adversarial_classifier = kwargs["adversarial_classifier"]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step."""
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )

        if optimizer_idx == 0:
            loss_output_objs = []
            n_obs = 0
            zs = []

            for i, tensors in enumerate(batch):
                n_obs += tensors[REGISTRY_KEYS.X_KEY].shape[0]
                self.loss_kwargs.update(
                    {"kl_weight": self.kl_weight, "mode": i}
                )
                inference_kwargs = {"mode": i}
                generative_kwargs = {"mode": i}

                inference_outputs, _, loss_output = self.forward(
                    tensors,
                    loss_kwargs=self.loss_kwargs,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                )

                _assert_finite(f"mode{i}/z", inference_outputs["z"])
                _assert_finite(f"mode{i}/qz.loc", inference_outputs["qz"].loc)
                _assert_finite(f"mode{i}/qz.scale", inference_outputs["qz"].scale)
                _assert_finite(f"mode{i}/loss", loss_output.loss)
                _assert_finite(
                    f"mode{i}/reconstruction_loss",
                    loss_output.reconstruction_loss,
                )
                _assert_finite(
                    f"mode{i}/kl_local",
                    loss_output.kl_local,
                )

                zs.append(inference_outputs["z"])
                loss_output_objs.append(loss_output)

            loss = sum(scl.loss for scl in loss_output_objs)
            loss /= max(int(n_obs), 1)

            rec_loss = sum(
                scl.reconstruction_loss_sum
                for scl in loss_output_objs
            )
            kl = sum(
                scl.kl_local_sum
                for scl in loss_output_objs
            )

            batch_tensor = [
                torch.zeros(
                    (z.shape[0], 1),
                    device=z.device,
                ) + i
                for i, z in enumerate(zs)
            ]

            if kappa > 0 and self.adversarial_classifier is not False:
                fool_loss = self.loss_adversarial_classifier(
                    torch.cat(zs),
                    torch.cat(batch_tensor),
                    False,
                )
                _assert_finite("adversarial/fool_loss", fool_loss)
                loss = loss + fool_loss * kappa

            _assert_finite("training/total_loss", loss)

            # Gradient clipping is handled by Lightning's Trainer through
            # gradient_clip_val in GIMVI_GCN.train().
            return {
                "loss": loss,
                "reconstruction_loss_sum": rec_loss,
                "kl_local_sum": kl,
                "kl_global": 0.0,
                "n_obs": n_obs,
            }

        if optimizer_idx == 1:
            zs = []
            for i, tensors in enumerate(batch):
                inference_inputs = self.module._get_inference_input(tensors)
                inference_inputs.update({"mode": i})
                outputs = self.module.inference(**inference_inputs)
                _assert_finite(f"classifier/mode{i}/z", outputs["z"])
                zs.append(outputs["z"])

            batch_tensor = [
                torch.zeros(
                    (z.shape[0], 1),
                    device=z.device,
                ) + i
                for i, z in enumerate(zs)
            ]

            loss = self.loss_adversarial_classifier(
                torch.cat(zs).detach(),
                torch.cat(batch_tensor),
                True,
            )
            loss = loss * kappa
            _assert_finite("adversarial/classifier_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Validation step."""
        self.loss_kwargs.update(
            {
                "kl_weight": self.kl_weight,
                "mode": dataloader_idx,
            }
        )
        inference_kwargs = {"mode": dataloader_idx}
        generative_kwargs = {"mode": dataloader_idx}

        inference_outputs, _, loss_output = self.forward(
            batch,
            loss_kwargs=self.loss_kwargs,
            inference_kwargs=inference_kwargs,
            generative_kwargs=generative_kwargs,
        )

        _assert_finite("validation/z", inference_outputs["z"])
        _assert_finite("validation/loss", loss_output.loss)

        reconstruction_loss = loss_output.reconstruction_loss_sum
        return {
            "reconstruction_loss_sum": reconstruction_loss,
            "kl_local_sum": loss_output.kl_local_sum,
            "kl_global": loss_output.kl_global,
            "n_obs": loss_output.n_obs_minibatch,
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_obs, elbo, rec_loss, kl_local = 0, 0, 0, 0

        for dl_out in outputs:
            for tensors in dl_out:
                elbo += (
                    tensors["reconstruction_loss_sum"]
                    + tensors["kl_local_sum"]
                )
                rec_loss += tensors["reconstruction_loss_sum"]
                kl_local += tensors["kl_local_sum"]
                n_obs += tensors["n_obs"]

        if n_obs == 0:
            return

        self.log("elbo_validation", elbo / n_obs)
        self.log(
            "reconstruction_loss_validation",
            rec_loss / n_obs,
        )
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", 0.0)
