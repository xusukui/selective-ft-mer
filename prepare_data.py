"""Repack official M3ED features, excluding audio of at most 0.2 seconds."""

import argparse
import os
import pickle

import soundfile as sf


ORIGINAL_PICKLE = (
    "Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-"
    "Lsent_cls_robert_wwm_base_chinese4chmed.pkl"
)


def get_audio_duration(audio_path):
    # Missing/unreadable files are input errors, not short utterances.
    info = sf.info(audio_path)
    if not info.samplerate:
        raise ValueError(f"Invalid sample rate: {audio_path}")
    return info.frames / info.samplerate


def prepare_features(original_features, audio_dir, min_audio_sec=0.2):
    """Preserve official splits, labels, order, text and feature arrays.

    Original IDs include A_/B_ prefixes used in WAV filenames.
    Refined sent_id omits that prefix, matching the research data.
    Field 2 contains integer labels corresponding to final_main_emo.
    """
    if min_audio_sec < 0:
        raise ValueError("min_audio_sec must be nonnegative")
    audio_dir = os.path.abspath(audio_dir)
    features, counts = {}, {}
    for pickle_idx, split in ((7, "train"), (8, "valid"), (9, "test")):
        features[split] = {}
        stats = {"original": 0, "retained": 0, "excluded_short": 0, "zero_frame": 0}
        for dialog_id in original_features[pickle_idx]:
            utterances = []
            for index, full_id in enumerate(original_features[0][dialog_id]):
                audio_id = full_id[5:] if full_id.startswith("m3ed_") else full_id
                sent_id = audio_id[2:] if audio_id.startswith(("A_", "B_")) else audio_id
                audio_path = os.path.join(audio_dir, audio_id + ".wav")
                duration = get_audio_duration(audio_path)
                stats["original"] += 1
                if duration <= min_audio_sec:
                    stats["excluded_short"] += 1
                    stats["zero_frame"] += int(duration == 0)
                    continue
                utterances.append({
                    "sent_id": sent_id,
                    "label": original_features[2][dialog_id][index],
                    "text_feature": original_features[3][dialog_id][index],
                    "audio_feature": original_features[4][dialog_id][index],
                    "vision_feature": original_features[5][dialog_id][index],
                    "text": original_features[6][dialog_id][index],
                    "audio_path": audio_path,
                    "audio_length": duration,
                })
                stats["retained"] += 1
            if utterances:
                features[split][dialog_id] = utterances
        counts[split] = stats
    return features, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feat_dir", required=True, help="Directory with the official feature pickle")
    parser.add_argument("--audio_dir", required=True, help="Directory containing A_*.wav / B_*.wav")
    parser.add_argument("--output", default="data/m3ed/m3ed_features.pkl")
    parser.add_argument("--min_audio_sec", type=float, default=0.2,
                        help="Exclude audio at or below this duration (seconds)")
    args = parser.parse_args()
    with open(os.path.join(args.feat_dir, ORIGINAL_PICKLE), "rb") as stream:
        original = pickle.load(stream)
    features, counts = prepare_features(original, args.audio_dir, args.min_audio_sec)
    for split, stats in counts.items():
        print(f"{split}: {len(features[split])} dialogues, {stats}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as stream:
        pickle.dump(features, stream)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
