"""Evaluate released .pt weights; optionally export dialogue-aligned predictions."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UnifiedEmotionModel
from data_loader import load_m3ed_data, load_meld_data


ENCODERS = {
    "m3ed": ("Johnson8187/Chinese-Emotion", "TencentGameMate/chinese-hubert-large"),
    "meld": ("FacebookAI/roberta-base", "facebook/wav2vec2-base"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Released .pt file")
    parser.add_argument("--dataset", required=True, choices=ENCODERS)
    parser.add_argument("--data_path", required=True, help="M3ED refined pickle or MELD base directory")
    parser.add_argument("--device", default=None, help="cuda or cpu (automatic if omitted)")
    parser.add_argument("--batch_size", type=int, choices=[1], default=1, help="One dialogue per batch")
    parser.add_argument("--text_model_path", help="Local directory or Hugging Face model ID")
    parser.add_argument("--audio_model_path", help="Local directory or Hugging Face model ID")
    parser.add_argument("--audio_dir", help="Optional M3ED directory replacing saved absolute audio paths")
    parser.add_argument("--output_predictions", help="Optional CSV with dialogue/utterance IDs and labels")
    return parser.parse_args()


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not all(
        key in checkpoint for key in ("model_state_dict", "hparams")
    ):
        raise ValueError("Expected a released .pt with model_state_dict and hparams.")
    if not isinstance(checkpoint["hparams"], dict) or "input_dims" not in checkpoint["hparams"]:
        raise ValueError("Checkpoint is missing its model configuration.")
    return checkpoint


def build_model(hparams, dataset, text_model_path=None, audio_model_path=None):
    default_text, default_audio = ENCODERS[dataset]
    # Use portable defaults, not the author's paths stored inside old checkpoints.
    return UnifiedEmotionModel(
        input_dims=hparams["input_dims"],
        modalities=hparams.get("modalities", ["text", "audio", "vision"]),
        use_cross_attn=hparams.get("use_cross_attn", True),
        use_self_attn=hparams.get("use_self_attn", True),
        causal_self_attn=hparams.get("causal_self_attn", False),
        fusion_arch=hparams.get("fusion_arch", "transformer"),
        fusion_depth=hparams.get("fusion_depth", 1),
        fusion_d_ff=hparams.get("fusion_d_ff", 1024),
        fusion_dropout=hparams.get("fusion_dropout", 0.1),
        d_model=hparams.get("d_model", 256),
        n_heads=hparams.get("n_heads", 8),
        num_classes=hparams.get("num_classes", 7),
        dynamic_text=hparams.get("dynamic_text", True),
        dynamic_audio=hparams.get("dynamic_audio", True),
        finetune_text=False,
        finetune_audio=False,
        text_model_path=text_model_path or default_text,
        audio_model_path=audio_model_path or default_audio,
    )


def load_checkpoint(model, checkpoint):
    """Only a wholly omitted frozen encoder may use pretrained weights."""
    state = checkpoint["model_state_dict"]
    config = checkpoint["hparams"]
    optional_prefixes = []
    for modality in ("text", "audio"):
        prefix = modality + "_enc."
        if not config.get("finetune_" + modality, False) and not any(
            key.startswith(prefix) for key in state
        ):
            optional_prefixes.append(prefix)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [key for key in missing if not key.startswith(tuple(optional_prefixes))]
    if critical or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={critical}, unexpected={unexpected}")
    return missing


def run_inference(model, dataloader, device):
    model.eval()
    all_preds, all_targets, rows = [], [], []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference"):
            outputs = model(batch["features"].to(device), batch["texts"], batch["audio_paths"])
            preds = outputs.argmax(dim=-1).cpu().numpy().reshape(-1)
            targets = batch["labels"].cpu().numpy().reshape(-1)
            sent_ids = [entry[0] for entry in batch["sent_ids"]]
            if not (len(preds) == len(targets) == len(sent_ids)):
                raise ValueError("Predictions, labels and utterance IDs must align.")
            dialog_id = batch["dialog_id"][0]
            rows.extend((dialog_id, sent_id, int(target), int(pred))
                        for sent_id, target, pred in zip(sent_ids, targets, preds))
            all_preds.extend(preds)
            all_targets.extend(targets)
    if not all_targets:
        raise ValueError("No test utterances were loaded.")
    return np.asarray(all_preds), np.asarray(all_targets), rows


def save_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dialog_id", "sent_id", "target", "prediction"])
        writer.writerows(rows)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = read_checkpoint(args.checkpoint)
    model = build_model(checkpoint["hparams"], args.dataset,
                        args.text_model_path, args.audio_model_path)
    load_checkpoint(model, checkpoint)
    del checkpoint
    model.to(device)
    if args.dataset == "m3ed":
        dataset = load_m3ed_data(args.data_path, "test", audio_dir=args.audio_dir)
    else:
        dataset = load_meld_data(args.data_path, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds, targets, rows = run_inference(model, loader, device)
    print(f"Dataset: {args.dataset}; dialogues: {len(dataset)}; utterances: {len(targets)}")
    print(f"Weighted Accuracy: {accuracy_score(targets, preds) * 100:.2f}%")
    print(f"Weighted F1: {f1_score(targets, preds, average='weighted', zero_division=0) * 100:.2f}%")
    if args.output_predictions:
        save_predictions(args.output_predictions, rows)
        print(f"Predictions saved to: {args.output_predictions}")


if __name__ == "__main__":
    main()
