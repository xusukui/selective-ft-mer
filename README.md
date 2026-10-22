# SelectiveFT-MER: Selective Fine-Tuning for Multimodal Emotion Recognition

Code and pretrained checkpoints for **Selective Encoder Fine-Tuning and Attention Fusion for Dialogue-Level Multimodal Emotion Recognition**.

The model combines pretrained text/audio encoders, fixed visual features, intra-modal self-attention and optional cross-modal attention.

## Reported results

| Dataset | Configuration | WA (%) | WF1 (%) |
| :---: | :---: | :---: | :---: |
| M³ED | SA + one-layer CA; audio fine-tuning | 61.2 | 58.0 |
| MELD | SA only; text fine-tuning | 64.5 | 62.5 |

## Installation

Use Python 3.12 and install the dependencies:

```bash
pip install -r requirements.txt
```

The checked environment uses PyTorch/torchaudio 2.7.1, Transformers 4.57.1 and SpeechBrain 1.0.3 on Linux. Install the matching PyTorch and torchaudio builds for your CPU or CUDA environment. The inference code processes one dialogue at a time.

## Encoders and checkpoints

The following pretrained encoders are loaded automatically from Hugging Face unless local paths are supplied:

| Dataset | Text encoder | Audio encoder |
| :---: | :---: | :---: |
| M³ED | [Johnson8187/Chinese-Emotion](https://huggingface.co/Johnson8187/Chinese-Emotion) | [TencentGameMate/chinese-hubert-large](https://huggingface.co/TencentGameMate/chinese-hubert-large) |
| MELD | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) |

Download the existing model packages from Baidu Drive:

| Dataset | Download | Access code |
| :---: | :---: | :---: |
| M³ED | [m3ed-checkpoint](https://pan.baidu.com/s/1OXZGWRmQiemKiNIlLd5Hbw?pwd=w6hn) | `w6hn` |
| MELD | [meld-checkpoint](https://pan.baidu.com/s/1ulLX6NcCHasYUgiLc7QtKg?pwd=6gy5) | `6gy5` |

The packages contain the existing `.pt` weights and YAML configuration. Put the weights at `checkpoints/m3ed_best.pt` and `checkpoints/meld_best.pt`, or pass their actual filenames to `--checkpoint`.

The `.pt` files already contain the model configuration and fine-tuned weights. Inference does not require a training log or a separate YAML file. Frozen encoders are loaded from their pretrained sources; both encoder resources are therefore needed. Local path arguments override the original paths saved in the checkpoints.

## Dataset preparation

### M³ED

Obtain the audio and original feature pickle from the [official M³ED repository](https://github.com/AIM3-RUC/RUCM3ED). The expected pickle is:

```text
Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl
```

Prepare the data:

```bash
python prepare_data.py \
    --feat_dir /path/to/m3ed/features \
    --audio_dir /path/to/m3ed/modality_speech \
    --output data/m3ed/m3ed_features.pkl
```

The audio directory contains filenames such as `A_fendou_1_1.wav`. Preparation preserves the official train/validation/test split, sample order, text, features and primary emotion labels corresponding to `final_main_emo`. Multi-label utterances are retained with their official primary label.

Audio of **at most 0.2 seconds**, including zero-frame WAVs, is excluded. Expected counts are:

| Split | Dialogues | Retained utterances | Excluded utterances |
| --- | ---: | ---: | ---: |
| train | 685 | 17,376 | 51 |
| valid | 126 | 2,814 | 7 |
| test | 179 | 4,180 | 21 |
| Total | 990 | 24,370 | 79 |

Missing or unreadable files raise an error. They are not silently counted as short audio. If a prepared pickle was moved between machines, use `--audio_dir` during inference to replace its saved audio directory.

### MELD

Obtain the [MELD dataset](https://affective-meld.github.io/), the [MELD_Text_Audio distribution](https://huggingface.co/datasets/zrr1999/MELD_Text_Audio), and the visual feature pickles referenced by [MM-Align](https://github.com/declare-lab/MM-Align).

```text
/path/to/meld/
├── MELD_Text_Audio/
│   ├── train_sent_emo.csv
│   ├── valid_sent_emo.csv
│   ├── test_sent_emo.csv
│   └── archive/
│       ├── train/dia0_utt0.flac
│       ├── valid/...
│       └── test/...
└── pickle_features/
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

The validation files/directories may also be named `dev`; the loader accepts both names, preferring `valid` when both exist. Visual pickle entries are keyed by `DialogueID_UtteranceID` and contain `video_features`; frame features are mean-pooled. Missing visual features raise an error.

The MELD loader retains the original filtering rule (audio shorter than 0.2 seconds is excluded). The checked data contain 9,879 / 1,093 / 2,576 utterances in train / validation / test; the test set has 280 dialogues.

## Inference

Run from the repository directory:

```bash
python inference.py \
    --checkpoint checkpoints/m3ed_best.pt \
    --dataset m3ed \
    --data_path data/m3ed/m3ed_features.pkl \
    --text_model_path /path/to/Chinese-Emotion \
    --audio_model_path /path/to/chinese-hubert-large
```

```bash
python inference.py \
    --checkpoint checkpoints/meld_best.pt \
    --dataset meld \
    --data_path /path/to/meld \
    --text_model_path /path/to/roberta-base \
    --audio_model_path /path/to/wav2vec2-base
```

Omit the encoder path arguments to use the Hugging Face model IDs. The device is selected automatically; use `--device cpu` or `--device cuda` to override it. Batch size is one dialogue.

The script prints WA (utterance accuracy) and weighted F1 calculated over all retained test utterances. It does not substitute metrics stored in old checkpoint metadata.

Add `--output_predictions results/meld_predictions.csv` (or an M³ED output filename) to export:

```text
dialog_id,sent_id,target,prediction
```

Label IDs follow these mappings:

- M³ED: Happy=0, Neutral=1, Sad=2, Disgust=3, Anger=4, Fear=5, Surprise=6.
- MELD: neutral=0, surprise=1, fear=2, sadness=3, joy=4, disgust=5, anger=6.

The export retains dialogue boundaries for downstream analysis. The manuscript uses dialogue-level resampling, so treating all utterances as independent would implement a different procedure.

## Training and optional weight export

Edit the data and encoder paths in the relevant YAML, then run:

```bash
python train.py hparams/m3ed_sa_ca_ft_audio.yaml
python train.py hparams/meld_sa_only_ft_text.yaml
```

The recipes use seed 1993, 30 epochs, batch size 1 dialogue, learning rates 1e-4 / 1e-5 for the main components / pretrained encoders, d_model=256 and 8 attention heads. New runs produce their own training logs and checkpoints. Selection uses the lowest validation error over all utterances.

To export a newly trained model into the existing `.pt` format:

```bash
python export_weights.py \
    --exp_dir results/m3ed_sb \
    --output checkpoints/my_m3ed_model.pt
```

For MELD use `results/meld_sb`. Export reads `hyperparams.yaml` and `save/CKPT+*/`; it works without a training log. Historical checkpoints do not need to be exported again.

## Citation

```bibtex
@unpublished{xu2026selective,
  title={Selective Encoder Fine-Tuning and Attention Fusion for Dialogue-Level Multimodal Emotion Recognition},
  author={Xu, Sukui and Zhu, Weijie and Pan, Jingmin},
  year={2026},
  note={Manuscript}
}
```

## License and external resources

Project code is distributed under the [MIT License](LICENSE). Datasets, visual features and pretrained encoders are obtained separately from the linked providers and remain subject to their respective terms. The project license does not relicense those resources.
