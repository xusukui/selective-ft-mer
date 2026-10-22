"""
Data loaders for M3ED and MELD datasets.

Supports both pre-extracted pickle features and dynamic encoding from raw audio/text.
"""

import os
import pickle
from typing import Dict, List, Optional

import pandas as pd
import soundfile as sf
import torch

# Emotion mappings
M3ED_EMOTION_MAP = {
    'Happy': 0, 'Neutral': 1, 'Sad': 2, 'Disgust': 3,
    'Anger': 4, 'Fear': 5, 'Surprise': 6
}

MELD_EMOTION_MAP = {
    'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3,
    'joy': 4, 'disgust': 5, 'anger': 6
}


class M3EDDialogDataset(torch.utils.data.Dataset):
    """Load M3ED dialog samples from pickle features.

    Args:
        data: Dictionary mapping dialog_id -> list of utterance dicts
    """
    def __init__(self, data, audio_dir=None):
        self.dialogue_ids = list(data.keys())
        self.data = data
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, idx):
        dialog_id = self.dialogue_ids[idx]
        dialog_data = self.data[dialog_id]

        features, labels = [], []
        texts = []
        audio_paths = []

        for utt in dialog_data:
            text_feat = torch.tensor(utt["text_feature"], dtype=torch.float32)
            audio_feat = torch.tensor(utt["audio_feature"], dtype=torch.float32)
            vision_feat = torch.tensor(utt["vision_feature"], dtype=torch.float32)
            fused_feat = torch.cat([text_feat, audio_feat, vision_feat], dim=-1)

            features.append(fused_feat.squeeze())
            labels.append(utt["label"])
            texts.append(str(utt["text"]))
            audio_path = utt["audio_path"]
            if self.audio_dir is not None:
                audio_path = os.path.join(self.audio_dir, os.path.basename(audio_path))
            audio_paths.append(audio_path)

        return {
            "dialog_id": str(dialog_id),
            "features": torch.stack(features),
            "texts": texts,
            "audio_paths": audio_paths,
            "labels": torch.tensor(labels),
            "sent_ids": [utt["sent_id"] for utt in dialog_data],
        }


class MELDDialogDataset(torch.utils.data.Dataset):
    """Load MELD dialog samples from CSV and audio files.

    Args:
        csv_file: Path to MELD CSV file (e.g., test_sent_emo.csv)
        archive_root: Root directory containing audio files
        split: Audio directory name ('train', 'valid' or 'dev', 'test')
        vision_pickle_file: Required pickle file with vision features
        vision_dim: Vision feature dimension
        min_audio_sec: Minimum audio duration in seconds
    """
    def __init__(
        self,
        csv_file: str,
        archive_root: str,
        split: str,
        vision_pickle_file: Optional[str] = None,
        vision_dim: int = 2048,
        min_audio_sec: float = 0.2,
    ):
        self.vision_dim = vision_dim
        self.min_audio_sec = min_audio_sec

        # Load vision features
        if not vision_pickle_file or not os.path.isfile(vision_pickle_file):
            raise FileNotFoundError(f"MELD visual features are required: {vision_pickle_file}")
        with open(vision_pickle_file, "rb") as stream:
            self.vision_pickle = pickle.load(stream)

        df = pd.read_csv(csv_file)

        entries = []
        split_dir = os.path.join(archive_root, split)
        for _, row in df.iterrows():
            did = int(row["Dialogue_ID"])
            uid = int(row["Utterance_ID"])
            sent_id = f"dia{did}_utt{uid}"
            audio_path = os.path.join(split_dir, f"{sent_id}.flac")

            dur = self._audio_duration_sec(audio_path)
            if dur < self.min_audio_sec:
                continue

            emo = str(row["Emotion"]).lower()
            label = MELD_EMOTION_MAP.get(emo)
            if label is None:
                raise ValueError(f"Unknown MELD emotion: {emo}")

            entries.append({
                "dialog_id": did,
                "utt_idx": uid,
                "sent_id": sent_id,
                "text": str(row["Utterance"]),
                "audio_path": audio_path,
                "label": label,
            })

        # Group by dialogue
        self.dialogues: Dict[int, List[Dict]] = {}
        for e in entries:
            self.dialogues.setdefault(e["dialog_id"], []).append(e)
        for d in self.dialogues.values():
            d.sort(key=lambda x: x["utt_idx"])

        self._index = list(self.dialogues.keys())

    def __len__(self):
        return len(self._index)

    def _audio_duration_sec(self, path: str):
        info = sf.info(path)
        if not info.samplerate:
            raise ValueError(f"Invalid sample rate: {path}")
        return float(info.frames) / float(info.samplerate)

    def _load_vision(self, sent_id: str) -> Optional[torch.Tensor]:
        pickle_key = sent_id.replace("dia", "").replace("utt", "").replace("_", "_")
        parts = sent_id.replace("dia", "").replace("utt", "").split("_")
        if len(parts) == 2:
            pickle_key = f"{parts[0]}_{parts[1]}"

        if self.vision_pickle and pickle_key in self.vision_pickle:
            video_feat = self.vision_pickle[pickle_key].get("video_features")
            if video_feat is not None:
                t = torch.as_tensor(video_feat, dtype=torch.float32)
                return t.mean(0) if t.ndim > 1 else t
        return None

    def __getitem__(self, idx):
        did = self._index[idx]
        utts = self.dialogues[did]

        features, labels, texts, audio_paths, sent_ids = [], [], [], [], []
        for u in utts:
            sent_id = u["sent_id"]

            # Only vision features - text/audio will be dynamically encoded in forward()
            vision_feat = self._load_vision(sent_id)
            if vision_feat is not None:
                features.append(vision_feat)
            else:
                raise ValueError(f"Missing MELD visual feature: {sent_id}")

            labels.append(u["label"])
            texts.append(u["text"])
            audio_paths.append(u["audio_path"])
            sent_ids.append(sent_id)

        return {
            "dialog_id": str(did),
            "features": torch.stack(features),
            "texts": texts,
            "audio_paths": audio_paths,
            "labels": torch.tensor(labels, dtype=torch.long),
            "sent_ids": sent_ids,
        }


def load_m3ed_data(pickle_path: str, split: str = "test", audio_dir=None):
    """Load M3ED data from pickle file.

    Args:
        pickle_path: Path to the M3ED pickle file
        split: Data split ('train', 'val', 'test')

    Returns:
        M3EDDialogDataset instance
    """
    with open(pickle_path, "rb") as f:
        features = pickle.load(f)
    candidates = ("valid", "dev", "val") if split in ("valid", "dev", "val") else (split,)
    for key in candidates:
        if key in features:
            return M3EDDialogDataset(features[key], audio_dir=audio_dir)
    raise KeyError(f"Missing split {split} in {pickle_path}")


def load_meld_data(base_dir: str, split: str = "test"):
    """Accept either valid or dev filenames for the validation split."""
    candidates = ("valid", "dev") if split in ("valid", "dev", "val") else (split,)

    def existing(root, suffix=""):
        for name in candidates:
            path = os.path.join(root, name + suffix)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Expected one of {candidates} under {root} (suffix {suffix!r})")

    csv_file = existing(os.path.join(base_dir, "MELD_Text_Audio"), "_sent_emo.csv")
    archive_root = os.path.join(base_dir, "MELD_Text_Audio", "archive")
    audio_split = os.path.basename(existing(archive_root))
    vision_pickle = existing(os.path.join(base_dir, "pickle_features"), ".pkl")
    return MELDDialogDataset(csv_file, archive_root, audio_split, vision_pickle)
