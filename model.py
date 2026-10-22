"""
Unified Emotion Recognition Model with Self-Attention and Cross-Modal Attention.

This module implements the model architecture described in:
"Selective Encoder Fine-Tuning and Attention Fusion for
Dialogue-Level Multimodal Emotion Recognition"
"""

import torch
from torch import nn
import torchaudio
import torchaudio.functional as AF
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor

# Default HuggingFace model paths
DEFAULT_TEXT_MODEL = "Johnson8187/Chinese-Emotion"
DEFAULT_AUDIO_MODEL = "TencentGameMate/chinese-hubert-large"


def load_mono_waveform(path, target_sr=16000, prefer_center_for_5p1=True):
    """Load audio file and convert to mono waveform.

    Args:
        path: Path to audio file
        target_sr: Target sample rate
        prefer_center_for_5p1: For 5.1 audio, use center channel (dialogue-focused)

    Returns:
        Tuple of (waveform as 1D numpy array, sample rate)
    """
    waveform, sr = torchaudio.load(path)
    ch = waveform.shape[0]

    if ch == 1:
        mono = waveform
    elif ch == 6 and prefer_center_for_5p1:
        # 5.1 layout: [FL, FR, FC, LFE, SL, SR], use center channel
        mono = waveform[2:3, :]
    else:
        mono = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        mono = AF.resample(mono, sr, target_sr)
        sr = target_sr

    return mono.squeeze(0).numpy(), sr


class SelfAttention(nn.Module):
    """Intra-modal Multi-Head Self-Attention with residual connection.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        causal: If True, apply causal masking for real-time inference
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, x):
        normed_x = self.ln(x)

        if self.causal:
            T = x.size(1)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            y, _ = self.attn(normed_x, x, x, attn_mask=causal_mask, need_weights=False)
        else:
            y, _ = self.attn(normed_x, x, x, need_weights=False)

        return x + self.drop(y)


class CrossModalProcessor(nn.Module):
    """Two-way attention from a target modality to two source modalities."""
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, target, source1, source2):
        out1, _ = self.attn1(target, source1, source1)
        out2, _ = self.attn2(target, source2, source2)
        return torch.cat([out1, out2], dim=-1)


class MultiModalFusion(nn.Module):
    """Symmetric cross-modal attention: every modality attends to the other two."""
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.proc_A = CrossModalProcessor(d_model, num_heads)
        self.proc_V = CrossModalProcessor(d_model, num_heads)
        self.proc_T = CrossModalProcessor(d_model, num_heads)

    def forward(self, T, A, V):
        fused_A = self.proc_A(A, V, T)
        fused_V = self.proc_V(V, T, A)
        fused_T = self.proc_T(T, V, A)
        return torch.cat([fused_A, fused_V, fused_T], dim=-1)


class CrossModalTFBlock(nn.Module):
    """A Transformer-style cross-modal block for Text, Audio, Vision."""
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.drop = nn.Dropout(dropout)

        # Text attends to (Audio, Vision)
        self.t_ln1 = nn.LayerNorm(d_model)
        self.t_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.t_ln2 = nn.LayerNorm(d_model)
        self.t_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

        # Audio attends to (Text, Vision)
        self.a_ln1 = nn.LayerNorm(d_model)
        self.a_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.a_ln2 = nn.LayerNorm(d_model)
        self.a_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

        # Vision attends to (Text, Audio)
        self.v_ln1 = nn.LayerNorm(d_model)
        self.v_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.v_ln2 = nn.LayerNorm(d_model)
        self.v_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def _one_modality(self, x, kv1, kv2, ln1, attn, ln2, ffn):
        if self.norm_first:
            y = attn(ln1(x), torch.cat([kv1, kv2], dim=1), torch.cat([kv1, kv2], dim=1), need_weights=False)[0]
            x = x + self.drop(y)
            y = ffn(ln2(x))
            x = x + y
        else:
            y = attn(x, torch.cat([kv1, kv2], dim=1), torch.cat([kv1, kv2], dim=1), need_weights=False)[0]
            x = ln1(x + self.drop(y))
            y = ffn(x)
            x = ln2(x + y)
        return x

    def forward(self, T, A, V):
        T = self._one_modality(T, A, V, self.t_ln1, self.t_attn, self.t_ln2, self.t_ffn)
        A = self._one_modality(A, T, V, self.a_ln1, self.a_attn, self.a_ln2, self.a_ffn)
        V = self._one_modality(V, T, A, self.v_ln1, self.v_attn, self.v_ln2, self.v_ffn)
        return T, A, V


class CrossModalTransformer(nn.Module):
    """Stack N CrossModalTFBlocks."""
    def __init__(self, d_model, n_heads, depth=2, d_ff=1024, dropout=0.1, norm_first=True):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalTFBlock(d_model, n_heads, d_ff, dropout, norm_first)
            for _ in range(depth)
        ])

    def forward(self, T, A, V):
        for layer in self.layers:
            T, A, V = layer(T, A, V)
        return torch.cat([T, A, V], dim=-1)


class UnifiedEmotionModel(nn.Module):
    """
    Unified configurable model for multimodal emotion recognition.

    Supports:
        - Unimodal (text / audio / vision)
        - Multimodal with concatenation
        - Multimodal with self-attention and cross-modal attention fusion

    Args:
        input_dims: List of [text_dim, audio_dim, vision_dim]
        modalities: List of modalities to use, e.g., ['text', 'audio', 'vision']
        use_cross_attn: Whether to apply cross-modal attention
        use_self_attn: Whether to apply intra-modal self-attention
        causal_self_attn: Use causal masking for real-time inference
        fusion_arch: Fusion architecture type ('transformer' or 'mha')
        fusion_depth: Number of cross-attention layers
        d_model: Model dimension after projection
        n_heads: Number of attention heads
        num_classes: Number of emotion classes
        dynamic_text: Load text encoder (e.g., RoBERTa) for raw text input
        dynamic_audio: Load audio encoder (e.g., HuBERT) for raw audio input
        finetune_text: Fine-tune text encoder
        finetune_audio: Fine-tune audio encoder
        text_model_path: HuggingFace model path for text encoder
        audio_model_path: HuggingFace model path for audio encoder
    """
    def __init__(
        self,
        input_dims,
        fusion_arch="mha",
        fusion_depth=2,
        fusion_d_ff=1024,
        fusion_dropout=0.1,
        modalities=('text', 'audio', 'vision'),
        use_cross_attn=True,
        use_self_attn=False,
        causal_self_attn=False,
        d_model=256,
        n_heads=8,
        num_classes=7,
        finetune_text=False,
        finetune_audio=False,
        dynamic_text=False,
        dynamic_audio=False,
        text_model_path=None,
        audio_model_path=None,
    ):
        super().__init__()
        self.config = {
            "input_dims": input_dims,
            "fusion_arch": fusion_arch,
            "fusion_depth": fusion_depth,
            "fusion_d_ff": fusion_d_ff,
            "fusion_dropout": fusion_dropout,
            "modalities": modalities,
            "use_cross_attn": use_cross_attn,
            "use_self_attn": use_self_attn,
            "causal_self_attn": causal_self_attn,
            "d_model": d_model,
            "n_heads": n_heads,
            "num_classes": num_classes,
            "finetune_text": finetune_text,
            "finetune_audio": finetune_audio,
            "dynamic_text": dynamic_text,
            "dynamic_audio": dynamic_audio,
            # Paths are strictly for loading, not needed for config reconstruction if weights are saved
            # but used to infer dim.
            "text_model_path": text_model_path,
            "audio_model_path": audio_model_path,
        }

        self.modalities = list(modalities)
        self.use_cross_attn = use_cross_attn and len(self.modalities) > 1
        self.use_self_attn = use_self_attn
        if self.use_cross_attn and len(self.modalities) != 3:
            raise ValueError("Cross-modal fusion requires text, audio and vision.")

        print(f"Using modalities: {self.modalities}")

        self.dynamic_text = dynamic_text and ('text' in self.modalities)
        self.dynamic_audio = dynamic_audio and ('audio' in self.modalities)
        self.fusion_arch = fusion_arch

        text_dim, audio_dim, vision_dim = input_dims

        # Pretrained encoders
        if self.dynamic_text:
            if text_model_path is None:
                text_model_path = DEFAULT_TEXT_MODEL
            self.text_tok = AutoTokenizer.from_pretrained(text_model_path)
            self.text_enc = AutoModel.from_pretrained(text_model_path)
            for p in self.text_enc.parameters():
                p.requires_grad = finetune_text
            self.text_dim = self.text_enc.config.hidden_size
        else:
            self.text_enc = None
            self.text_dim = text_dim

        if self.dynamic_audio:
            if audio_model_path is None:
                audio_model_path = DEFAULT_AUDIO_MODEL
            self.audio_feat = AutoFeatureExtractor.from_pretrained(audio_model_path)
            self.audio_enc = AutoModel.from_pretrained(audio_model_path)
            for p in self.audio_enc.parameters():
                p.requires_grad = finetune_audio
            self.audio_dim = self.audio_enc.config.hidden_size
        else:
            self.audio_enc = None
            self.audio_dim = audio_dim

        self.vision_dim = vision_dim

        # Projections
        self.text_proj = nn.Identity()
        self.audio_proj = nn.Identity()
        self.vision_proj = nn.Identity()

        if 'text' in self.modalities:
            self.text_proj = nn.Sequential(
                nn.TransformerEncoderLayer(self.text_dim, nhead=8),
                nn.Linear(self.text_dim, d_model),
            )
        if 'audio' in self.modalities:
            self.audio_proj = nn.Sequential(
                nn.Linear(self.audio_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
        if 'vision' in self.modalities:
            self.vision_proj = nn.Sequential(
                nn.Linear(self.vision_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )

        # Self-attention
        if self.use_self_attn:
            self.sa_text = SelfAttention(d_model, n_heads, causal=causal_self_attn) if 'text' in self.modalities else None
            self.sa_audio = SelfAttention(d_model, n_heads, causal=causal_self_attn) if 'audio' in self.modalities else None
            self.sa_vision = SelfAttention(d_model, n_heads, causal=causal_self_attn) if 'vision' in self.modalities else None
        else:
            self.sa_text = self.sa_audio = self.sa_vision = None

        # Cross-modal fusion
        if self.use_cross_attn:
            if fusion_arch == "transformer":
                self.cross_attn = CrossModalTransformer(
                    d_model, n_heads, fusion_depth, fusion_d_ff, fusion_dropout
                )
            elif fusion_arch == "mha":
                self.cross_attn = MultiModalFusion(d_model, n_heads)
            else:
                raise ValueError(f"Unknown fusion_arch: {fusion_arch}")
        else:
            self.cross_attn = None

        # Classifier
        sa_dim = d_model * len(self.modalities) if self.use_self_attn else 0
        ca_dim = d_model * (6 if fusion_arch == "mha" else 3) if self.use_cross_attn else 0
        concat_dim = d_model * len(self.modalities)

        if self.use_cross_attn:
            clf_in = ca_dim
        else:
            clf_in = concat_dim

        if self.use_self_attn:
            clf_in += sa_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def _encode_text(self, texts, device):
        """Encode raw text using pretrained text encoder."""
        flat_texts = [t for dialog in texts for t in dialog]
        inputs = self.text_tok(flat_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        feat = self.text_enc(**inputs).last_hidden_state.mean(1)
        return feat.unsqueeze(0)

    def _encode_audio(self, audio_paths, device):
        """Encode raw audio using pretrained audio encoder."""
        audio_feats = []
        for path in audio_paths:
            waveform, sr = load_mono_waveform(path[0])
            inputs = self.audio_feat(waveform, sampling_rate=sr, return_tensors='pt').to(device)
            feat = self.audio_enc(**inputs).last_hidden_state.mean(1)
            audio_feats.append(feat)
        return torch.cat(audio_feats).unsqueeze(0)

    def forward(self, x, texts=None, audio_paths=None, return_features=False):
        """
        Forward pass.

        Args:
            x: Pre-extracted features tensor [B, T, D] containing concatenated
               (text, audio, vision) features. D = text_dim + audio_dim + vision_dim.
            texts: Raw text strings (used when dynamic_text=True)
            audio_paths: Paths to audio files (used when dynamic_audio=True)
            return_features: If True, return intermediate features for visualization

        Returns:
            Logits [B*T, num_classes] or tuple (logits, features) if return_features=True
        """
        device = x.device

        # Text
        if 'text' in self.modalities:
            if self.dynamic_text:
                text_feat = self._encode_text(texts, device)
            else:
                text_feat = x[..., :self.text_dim]
            text_proj = self.text_proj(text_feat)
            text_sa = self.sa_text(text_proj) if self.sa_text else None
        else:
            text_proj = None
            text_sa = None

        # Audio
        if 'audio' in self.modalities:
            if self.dynamic_audio:
                audio_feat = self._encode_audio(audio_paths, device)
            else:
                offset = 0 if 'text' not in self.modalities else self.text_dim
                audio_feat = x[..., offset:offset + self.audio_dim]
            audio_proj = self.audio_proj(audio_feat)
            audio_sa = self.sa_audio(audio_proj) if self.sa_audio else None
        else:
            audio_proj = None
            audio_sa = None

        # Vision
        if 'vision' in self.modalities:
            # Vision features are always at the end of the concatenated input
            vision_feat = x[..., -self.vision_dim:]
            vision_proj = self.vision_proj(vision_feat)
            vision_sa = self.sa_vision(vision_proj) if self.sa_vision else None
        else:
            vision_proj = None
            vision_sa = None

        # Cross-modal fusion
        if self.use_cross_attn:
            ca_out = self.cross_attn(text_proj, audio_proj, vision_proj)
        else:
            ca_out = None

        # Combine features
        # Combine features
        if ca_out is not None:
            fused = ca_out
        else:
            # If no cross-attn, fallback to concatenation of projections
            valid_projs = [p for p in (text_proj, audio_proj, vision_proj) if p is not None]
            fused = torch.cat(valid_projs, dim=-1)

        # Append self-attention features if enabled
        if self.use_self_attn:
            valid_sa = [s for s in (text_sa, audio_sa, vision_sa) if s is not None]
            if valid_sa:
                sa_feat = torch.cat(valid_sa, dim=-1)
                fused = torch.cat([fused, sa_feat], dim=-1)

        B, T_len, D = fused.shape
        fused_flat = fused.view(B * T_len, D)
        logits = self.classifier(fused_flat)

        if return_features:
            return logits, fused
        return logits
