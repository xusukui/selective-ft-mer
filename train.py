"""
SpeechBrain-based training script for multimodal emotion recognition.

Usage:
    python train.py hparams/m3ed_sa_ca_ft_audio.yaml
    python train.py hparams/meld_sa_only_ft_text.yaml
"""

import os
import sys
import pickle

from hyperpyyaml import load_hyperpyyaml
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.utils import metric_stats
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.distributed import ddp_init_group

from data_loader import load_m3ed_data, load_meld_data


class EmotionRecBrain(sb.Brain):
    """SpeechBrain Brain class for multimodal emotion recognition."""

    def compute_forward(self, batch, stage):
        """Forward pass through the model."""
        outputs = self.hparams.log_softmax(
            self.modules.mm_model(
                batch["features"].to(self.device),
                batch["texts"],
                batch["audio_paths"],
            )
        )
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss and collect predictions for metrics."""
        labels = batch["labels"].to(self.device).view(-1)
        loss = self.hparams.compute_cost(predictions, labels)

        if stage != sb.Stage.TRAIN:
            preds = predictions.argmax(dim=-1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            self.all_preds.append(preds)
            self.all_targets.append(targets)
            self.error_metrics.append(batch["sent_ids"], predictions, labels)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Initialize metrics at stage start."""
        self.loss_metric = metric_stats.MetricStats(metric=nll_loss)
        self.all_preds, self.all_targets = [], []

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Log metrics and save checkpoints at stage end."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            stats = None
        else:
            # Compute metrics
            final_preds = np.concatenate(self.all_preds).flatten()
            final_targets = np.concatenate(self.all_targets).flatten()
            weighted_f1 = f1_score(final_targets, final_preds, average="weighted")
            accuracy = (final_preds == final_targets).mean()

            stats = {
                "loss": stage_loss,
                "error_rate": float(1.0 - accuracy),
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
            }
            self.all_preds, self.all_targets = [], []

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "LR": self.optimizer.param_groups[0]["lr"]},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error_rate"])

        if stage == sb.Stage.TEST:
            self.test_stats = stats
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        """
        Build optimizer with separate learning rates for pretrained encoders.

        Parameter groups:
          1) Main: projections, cross_attn, self_attn, classifier
          2) Pretrained: text_enc, audio_enc (smaller learning rate)
        """
        model = self.modules.mm_model

        def trainable_params(m):
            if m is None:
                return []
            return [p for p in m.parameters() if p.requires_grad]

        main_params = []
        pretrained_params = []

        # Projections
        for name in ["text_proj", "audio_proj", "vision_proj"]:
            if hasattr(model, name):
                main_params += trainable_params(getattr(model, name))

        # Self-attention modules
        for name in ["sa_text", "sa_audio", "sa_vision"]:
            if hasattr(model, name) and getattr(model, name) is not None:
                main_params += trainable_params(getattr(model, name))

        # Cross-attention
        if getattr(model, "use_cross_attn", False) and hasattr(model, "cross_attn"):
            main_params += trainable_params(model.cross_attn)

        # Classifier
        if hasattr(model, "classifier"):
            main_params += trainable_params(model.classifier)

        # Pretrained encoders
        if hasattr(model, "text_enc") and model.text_enc is not None:
            pretrained_params += trainable_params(model.text_enc)
        if hasattr(model, "audio_enc") and model.audio_enc is not None:
            pretrained_params += trainable_params(model.audio_enc)

        # Build param groups
        opt_params = []
        base_lr = getattr(self.hparams, "lr", 1e-4)
        pretrained_lr = getattr(self.hparams, "pretrained_lr", base_lr * 0.1)
        weight_decay = getattr(self.hparams, "weight_decay", 0.0)

        if main_params:
            opt_params.append({
                "params": main_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        if pretrained_params:
            opt_params.append({
                "params": pretrained_params,
                "lr": pretrained_lr,
                "weight_decay": weight_decay,
            })

        # Fallback if everything is frozen
        if not opt_params:
            opt_params.append({"params": trainable_params(model), "lr": base_lr})

        self.optimizer = self.hparams.opt_class(opt_params)
        self.checkpointer.add_recoverable("optimizer", self.optimizer)


def load_dataset(hparams, split):
    """Load dataset based on hparams configuration."""
    dataset_name = hparams.get("dataset_name", "m3ed")

    if dataset_name == "m3ed":
        return load_m3ed_data(hparams["m3ed_feature_pickle"], split,
                             audio_dir=hparams.get("m3ed_audio_dir"))

    elif dataset_name == "meld":
        return load_meld_data(hparams["meld_base_dir"], split)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    ddp_init_group(run_opts)

    # Load hyperparameters
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Load datasets
    train_dataset = load_dataset(hparams, "train")
    valid_dataset = load_dataset(hparams, "valid")
    test_dataset = load_dataset(hparams, "test")

    batch_size = hparams.get("batch_size", 1)
    if batch_size != 1:
        raise ValueError("This dialogue-level recipe requires batch_size: 1.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} dialogues")
    print(f"Valid: {len(valid_dataset)} dialogues")
    print(f"Test: {len(test_dataset)} dialogues")

    # Initialize Brain
    brain = EmotionRecBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_loader,
        valid_set=valid_loader,
    )

    # Evaluate on test set
    brain.evaluate(test_set=test_loader, min_key="error_rate")

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    if hasattr(brain, "test_stats") and brain.test_stats:
        print(f"  Accuracy:    {brain.test_stats['accuracy']:.4f}")
        print(f"  Weighted F1: {brain.test_stats['weighted_f1']:.4f}")
        print(f"  Error Rate:  {brain.test_stats['error_rate']:.4f}")
    print("=" * 60)
