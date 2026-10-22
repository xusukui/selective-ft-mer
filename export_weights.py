"""
Export SpeechBrain checkpoint to minimal inference format using HyperPyYAML.

This version uses hyperpyyaml to robustly parse the hyperparameters file,
avoiding regex limitations. It requires speechbrain and hyperpyyaml.

Usage:
    python export_weights.py \
        --exp_dir results/m3ed/best_cfg \
        --output checkpoints/m3ed_best.pt
"""

import argparse
import os
import glob
import sys
import torch
import yaml
import hyperpyyaml

def parse_args():
    parser = argparse.ArgumentParser(description="Export SpeechBrain checkpoint to minimal format (Robust)")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to SpeechBrain experiment directory (containing save/ and hyperparams.yaml)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for exported weights (.pt)")
    parser.add_argument("--include_audio_enc", action="store_true",
                        help="Include fine-tuned audio encoder weights (auto-detected from hparams)")
    parser.add_argument("--include_text_enc", action="store_true",
                        help="Include fine-tuned text encoder weights (auto-detected from hparams)")
    return parser.parse_args()


def find_best_checkpoint(save_dir: str) -> tuple:
    """Find the best checkpoint based on error_rate in CKPT.yaml.

    Returns:
        Tuple of (checkpoint_dir, ckpt_meta)
    """
    ckpt_dirs = glob.glob(os.path.join(save_dir, "CKPT+*"))

    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {save_dir}")

    best_dir = None
    best_error = float("inf")
    best_meta = {}

    for ckpt_dir in ckpt_dirs:
        ckpt_yaml = os.path.join(ckpt_dir, "CKPT.yaml")
        if not os.path.exists(ckpt_yaml):
            continue

        with open(ckpt_yaml, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        error_rate = meta.get("error_rate", float("inf"))
        if error_rate < best_error:
            best_error = error_rate
            best_dir = ckpt_dir
            best_meta = meta

    if best_dir is None:
        raise FileNotFoundError(f"No valid checkpoints found in {save_dir}")

    return best_dir, best_meta


def parse_train_log(log_path: str) -> dict:
    """Parse train_log.txt to extract test results from the last line.

    Expected format:
        Epoch loaded: X - test loss: Y, test error_rate: Z, test weighted_f1: W
    """
    import re

    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find test line (usually last non-empty line)
    test_line = None
    for line in reversed(lines):
        if "test error_rate" in line.lower() or "test_error_rate" in line.lower():
            test_line = line
            break

    if not test_line:
        return {}

    results = {}

    # Parse test error_rate
    match = re.search(r"test[_ ]error[_ ]rate:\s*([\d.e+-]+)", test_line, re.IGNORECASE)
    if match:
        results["test_error_rate"] = float(match.group(1))
        results["test_wa"] = 1.0 - results["test_error_rate"]

    # Parse test weighted_f1
    match = re.search(r"test[_ ]weighted[_ ]f1:\s*([\d.e+-]+)", test_line, re.IGNORECASE)
    if match:
        results["test_wf1"] = float(match.group(1))

    return results


def EXTRACT_HPARAMS_FROM_MODULES(modules_dict):
    """Recursively search for UnifiedEmotionModel in modules dict."""
    for name, module in modules_dict.items():
        # Check if it's our target model
        # Use class name string check or instance check if model imported
        if hasattr(module, "config"):
            return module.config
    return {}


def load_hparams_via_hyperpyyaml(yaml_path: str) -> dict:
    """Load hyperparameters using hyperpyyaml.

    This instantiates the objects defined in the yaml.
    We then look for the 'modules' key or 'mm_model' key to find the UnifiedEmotionModel instance,
    and extract its 'config' attribute which we recently added to model.py.
    """

    # Ensure current dir is in sys.path
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    print(f"Parsing YAML with hyperpyyaml: {yaml_path}")
    with open(yaml_path, "r") as f:
        # hyperpyyaml.load_hyperpyyaml returns a dictionary where objects are instantiated
        hparams_dict = hyperpyyaml.load_hyperpyyaml(f)

    # Attempt to find the model config
    config = {}

    # 1. Check if 'mm_model' is directly in top-level
    if "mm_model" in hparams_dict and hasattr(hparams_dict["mm_model"], "config"):
        print("Found model config in 'mm_model' object")
        config = hparams_dict["mm_model"].config

    # 2. Check if it's in 'modules' dict
    elif "modules" in hparams_dict and isinstance(hparams_dict["modules"], dict):
        if "mm_model" in hparams_dict["modules"] and hasattr(hparams_dict["modules"]["mm_model"], "config"):
            print("Found model config in 'modules.mm_model'")
            config = hparams_dict["modules"]["mm_model"].config
        else:
             # Search any module
             print("Searching modules for UnifiedEmotionModel...")
             config = EXTRACT_HPARAMS_FROM_MODULES(hparams_dict["modules"])

    if not config:
        print("WARNING: Could not find UnifiedEmotionModel config in parsed YAML.")
        print("Falling back to extracting top-level keys if available (sanity check).")
        # Try to rescue some known keys if they exist as scalar values in yaml
        known_keys = ["d_model", "input_dims", "modalities", "num_classes"]
        for k in known_keys:
            if k in hparams_dict and not isinstance(hparams_dict[k], object):
                 config[k] = hparams_dict[k]

    return config


def extract_weights(state_dict: dict, include_audio_enc: bool, include_text_enc: bool) -> dict:
    """Extract relevant weights from state dict."""
    new_state_dict = {}

    for key, value in state_dict.items():
        # Always exclude tokenizer and feature extractor
        if key.startswith("text_tok.") or key.startswith("audio_feat."):
            continue

        # Handle encoder weights based on flags
        if key.startswith("audio_enc."):
            if include_audio_enc:
                new_state_dict[key] = value
        elif key.startswith("text_enc."):
            if include_text_enc:
                new_state_dict[key] = value
        else:
            # Include all other weights
            new_state_dict[key] = value

    return new_state_dict


def main():
    args = parse_args()

    # Find hyperparams.yaml
    hparams_path = os.path.join(args.exp_dir, "hyperparams.yaml")
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hyperparams.yaml not found in {args.exp_dir}")

    # Find best checkpoint
    save_dir = os.path.join(args.exp_dir, "save")
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"save/ directory not found in {args.exp_dir}")

    print(f"Searching for best checkpoint in: {save_dir}")
    best_ckpt_dir, ckpt_meta = find_best_checkpoint(save_dir)
    print(f"Found best checkpoint: {os.path.basename(best_ckpt_dir)}")
    print(f"  error_rate: {ckpt_meta.get('error_rate', 'N/A')}")
    print(f"  weighted_f1: {ckpt_meta.get('weighted_f1', 'N/A')}")

    # Load checkpoint
    model_ckpt_path = os.path.join(best_ckpt_dir, "model.ckpt")
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"model.ckpt not found in {best_ckpt_dir}")

    print(f"\nLoading checkpoint: {model_ckpt_path}")
    state_dict = torch.load(model_ckpt_path, map_location="cpu", weights_only=True)

    # Load hyperparameters via hyperpyyaml
    # This involves instantiating the model, so it might take a moment if it loads large pre-trained weights.
    # But it guarantees correctness.
    hparams = load_hparams_via_hyperpyyaml(hparams_path)

    if not hparams:
        raise ValueError("Failed to extract hyperparameters using hyperpyyaml. Check if model.py was updated to store self.config.")

    print(f"\nExtracted hparams: {hparams.keys()}")

    # Auto-detect encoder inclusion
    include_audio_enc = args.include_audio_enc or hparams.get("finetune_audio", False)
    include_text_enc = args.include_text_enc or hparams.get("finetune_text", False)

    print(f"\nInclude audio encoder: {include_audio_enc}")
    print(f"Include text encoder: {include_text_enc}")

    # Extract weights
    extracted = extract_weights(state_dict, include_audio_enc, include_text_enc)

    print(f"\nOriginal keys: {len(state_dict)}")
    print(f"Extracted keys: {len(extracted)}")

    # Parse test results from train_log.txt
    train_log_path = os.path.join(args.exp_dir, "train_log.txt")
    if os.path.exists(train_log_path):
        print(f"\nParsing test results from: {train_log_path}")
        meta = parse_train_log(train_log_path)
        if meta:
            print(f"  Test WA: {meta.get('test_wa', 0)*100:.2f}%")
            print(f"  Test WF1: {meta.get('test_wf1', 0)*100:.2f}%")
        else:
            print("  No test results found in train_log.txt")
            meta = {}
    else:
        print("\nNo training log; optional metrics metadata will be omitted.")
        meta = {}

    # Create output checkpoint
    output_ckpt = {
        "model_state_dict": extracted,
        "hparams": hparams,
        "meta": meta,
    }

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(output_ckpt, args.output)

    # Report
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Exported checkpoint saved to: {args.output}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"{'='*60}")
    print(f"Logged Test WA: {meta['test_wa']*100:.2f}%" if meta.get('test_wa') is not None else "Logged Test WA: N/A")
    print(f"Logged Test WF1: {meta['test_wf1']*100:.2f}%" if meta.get('test_wf1') is not None else "Logged Test WF1: N/A")


if __name__ == "__main__":
    main()
