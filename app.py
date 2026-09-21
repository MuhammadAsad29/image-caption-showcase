import os
import json
import time
import html
from io import BytesIO
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import streamlit as st

# =============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & INITIALIZATION
# =============================================================================
st.set_page_config(
    page_title="Multimodal Captioning — Dataset Scaling Study",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "last_results" not in st.session_state:
    st.session_state.last_results = {}
if "last_run_signature" not in st.session_state:
    st.session_state.last_run_signature = None
if "mode" not in st.session_state:
    st.session_state.mode = "Compare All 3"
if "beam_width" not in st.session_state:
    st.session_state.beam_width = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")

# Benchmark metadata from empirical research evaluation
BENCHMARKS = {
    "flickr8k": {
        "dataset_name": "Flickr8k",
        "dataset_size": "8,091 images (~40k captions)",
        "arch_label": "ResNet-50 + Plain LSTM",
        "decoding": "Greedy (Argmax)",
        "bleu4": "0.1819",
        "cider": "0.4457",
        "accent": "#f59e0b",
        "tag": "Small Scale Baseline"
    },
    "flickr30k": {
        "dataset_name": "Flickr30k",
        "dataset_size": "31,783 images (~158k captions)",
        "arch_label": "ResNet-50 + Bahdanau Attention",
        "decoding": "Beam Search",
        "bleu4": "0.2226",
        "cider": "0.4737",
        "accent": "#06b6d4",
        "tag": "Mid Scale Attention"
    },
    "coco": {
        "dataset_name": "MS COCO 2017",
        "dataset_size": "70,000+ subset (~350k captions)",
        "arch_label": "ResNet-50 + Bahdanau Attention",
        "decoding": "Beam Search",
        "bleu4": "0.2974",
        "cider": "0.9512",
        "accent": "#8b5cf6",
        "tag": "Large Scale SOTA"
    }
}


# =============================================================================
# HELPER: SAFE HTML RENDERING (PREVENTS MARKDOWN INDENTED CODE BLOCK LEAKS)
# =============================================================================

def render_html(html_str: str):
    """
    Renders raw HTML safely in Streamlit without triggering Markdown indented-code-block parsing.
    Strips leading and trailing indentation from each line, removes blank lines,
    and calls st.markdown(cleaned_html, unsafe_allow_html=True).
    """
    cleaned_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


# =============================================================================
# 2. MODEL ARCHITECTURE DEFINITIONS
# =============================================================================

class VisualEncoder(nn.Module):
    """
    Unified ResNet-50 Visual Encoder with dual extraction heads:
      - extract_pooled: (batch, 2048) pooled feature vector for Flickr8k (no attention)
      - extract_spatial: (batch, 49, 2048) spatial feature grid for Flickr30k & MS COCO (attention)
    """
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        
        # Spatial features: remove avgpool and fc -> keep conv5 feature map
        modules = list(resnet.children())[:-2]
        self.conv_features = nn.Sequential(*modules)
        self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Pooled features: remove fc layer -> keep avgpool (1x1)
        self.pooled_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze encoder parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.eval()

    def extract_pooled(self, images: torch.Tensor) -> torch.Tensor:
        """Returns flattened feature vector of shape (batch, 2048)"""
        feat = self.pooled_features(images)  # (batch, 2048, 1, 1)
        return feat.squeeze(-1).squeeze(-1)  # (batch, 2048)

    def extract_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Returns spatial region features of shape (batch, 49, 2048)"""
        features = self.conv_features(images)          # (batch, 2048, H, W)
        features = self.spatial_pool(features)         # (batch, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)        # (batch, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch, 49, 2048)
        return features


class PlainLSTMDecoder(nn.Module):
    """
    Plain LSTM Decoder (Flickr8k) without attention mechanism.
    The projected image feature is fed as the embedding at t=0, followed by
    embedded caption tokens at t=1..T.
    """
    def __init__(self, feature_dim: int, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.feature_proj = nn.Linear(feature_dim, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.eval()


class Attention(nn.Module):
    """
    Bahdanau-style Additive Soft Attention over 49 spatial visual regions.
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_out: (batch, 49, encoder_dim)
        # decoder_hidden: (batch, hidden_size)
        att1 = self.encoder_att(encoder_out)                    # (batch, 49, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)    # (batch, 1, att_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, 49)
        alpha = self.softmax(att)                               # (batch, 49)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch, encoder_dim)
        return context, alpha


class AttentionLSTMDecoder(nn.Module):
    """
    LSTMCell Decoder with Bahdanau Soft Attention (Flickr30k & MS COCO).
    Architecture:
      - Bahdanau Attention module
      - Embedding(vocab_size, embed_size, padding_idx=0)
      - init_h, init_c: Linear projections from mean spatial features to initial states
      - LSTMCell(embed_size + encoder_dim, hidden_size)
      - Dropout(0.6)
      - Linear(hidden_size, vocab_size)
    """
    def __init__(self, embed_size: int, hidden_size: int, attention_dim: int, vocab_size: int,
                 encoder_dim: int = 2048, dropout: float = 0.6):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.eval()

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c


# =============================================================================
# 3. INFERENCE DECODING STRATEGIES
# =============================================================================

# NOTE ON CACHING & INFERENCE PURITY (ROOT CAUSE AUDIT):
# Neither decode_greedy_plain, decode_beam_search_attn, nor run_caption_inference
# are decorated with @st.cache_data / @st.cache_resource. They execute freshly on every call,
# allowing beam_width variations (e.g. k=1 greedy vs k=5 vs k=10) to compute genuine outputs.

@torch.no_grad()
def decode_greedy_plain(feature: torch.Tensor, decoder: PlainLSTMDecoder,
                        word2idx: Dict[str, int], idx2word: Dict[Any, str],
                        max_len: int) -> str:
    """
    Greedy argmax decoding for Flickr8k Plain LSTM decoder.
    Feeds projected image feature at t=0, then iteratively samples the top token.
    """
    device = next(decoder.parameters()).device
    if not isinstance(feature, torch.Tensor):
        feature = torch.tensor(feature, dtype=torch.float32)
    if feature.dim() == 1:
        feature = feature.unsqueeze(0)
    feature = feature.to(device)

    feat_embed = decoder.feature_proj(feature).unsqueeze(1)  # (1, 1, embed_size)
    hidden = None
    inputs = feat_embed
    generated_ids = []

    for _ in range(max_len):
        lstm_out, hidden = decoder.lstm(inputs, hidden)
        output = decoder.fc(lstm_out.squeeze(1))
        predicted_id = output.argmax(dim=1).item()

        word_str = idx2word.get(str(predicted_id), idx2word.get(predicted_id, ""))
        if word_str == "endseq":
            break
        generated_ids.append(predicted_id)

        next_embed = decoder.embed(torch.tensor([[predicted_id]], device=device))
        inputs = next_embed

    words = [idx2word.get(str(i), idx2word.get(i, "")) for i in generated_ids]
    words = [w for w in words if w not in ("startseq", "endseq", "<pad>", "<unk>") and w != ""]
    return " ".join(words)


@torch.no_grad()
def decode_beam_search_attn(feature: torch.Tensor, decoder: AttentionLSTMDecoder,
                            word2idx: Dict[str, int], idx2word: Dict[Any, str],
                            max_len: int, beam_width: int = 5) -> str:
    """
    Length-normalized Beam Search decoding for Attention LSTM decoder (Flickr30k / COCO).
    Tracks live candidate beams, accumulates log-probabilities, and normalizes
    scores by candidate sequence length.

    When beam_width=1, this operates as exact greedy decoding (selecting top-1 at each step).
    When beam_width > 1, it maintains k simultaneous hypothesis paths.
    """
    device = next(decoder.parameters()).device
    if not isinstance(feature, torch.Tensor):
        feature = torch.tensor(feature, dtype=torch.float32)
    if feature.dim() == 2:
        feature = feature.unsqueeze(0)  # (1, 49, 2048)
    feature = feature.to(device)

    k = max(1, int(beam_width))
    start_id = word2idx["startseq"]
    end_id = word2idx["endseq"]
    vocab_size = decoder.fc.out_features

    h, c = decoder.init_hidden_state(feature)

    feature_k = feature.expand(k, -1, -1)
    h = h.expand(k, -1).contiguous()
    c = c.expand(k, -1).contiguous()

    seqs = torch.full((k, 1), start_id, dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, 1, device=device)

    complete_seqs = []
    complete_seqs_scores = []

    step = 1
    live_k = k

    while True:
        word = seqs[:, -1]
        embed = decoder.embed(word)
        context, _ = decoder.attention(feature_k[:live_k], h)
        lstm_input = torch.cat([embed, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(live_k, 0, True, True)

        prev_beam_idx = top_k_words // vocab_size
        next_word_idx = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_beam_idx], next_word_idx.unsqueeze(1)], dim=1)

        incomplete_idx = [i for i, w in enumerate(next_word_idx) if w.item() != end_id]
        complete_idx = [i for i in range(len(next_word_idx)) if i not in incomplete_idx]

        if len(complete_idx) > 0:
            complete_seqs.extend(seqs[complete_idx].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_idx].tolist())

        live_k -= len(complete_idx)
        if live_k == 0 or step >= max_len:
            if live_k > 0:
                complete_seqs.extend(seqs[incomplete_idx].tolist())
                complete_seqs_scores.extend(top_k_scores[incomplete_idx].tolist())
            break

        seqs = seqs[incomplete_idx]
        h = h[prev_beam_idx[incomplete_idx]]
        c = c[prev_beam_idx[incomplete_idx]]
        feature_k = feature_k[:live_k]
        top_k_scores = top_k_scores[incomplete_idx].unsqueeze(1)

        step += 1

    if not complete_seqs:
        return ""

    # Length-normalized selection
    best_idx = max(range(len(complete_seqs_scores)),
                   key=lambda i: complete_seqs_scores[i] / max(1, len(complete_seqs[i])))
    best_seq = complete_seqs[best_idx]

    words = [idx2word.get(str(i), idx2word.get(i, "")) for i in best_seq]
    words = [w for w in words if w not in ("startseq", "endseq", "<pad>", "<unk>") and w != ""]
    return " ".join(words)


# =============================================================================
# 4. RESOURCE CACHING & MODEL LOADER
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_visual_encoder() -> VisualEncoder:
    """Loads and caches the shared ResNet-50 visual encoder."""
    encoder = VisualEncoder().to(DEVICE)
    encoder.eval()
    return encoder


@st.cache_resource(show_spinner=False)
def load_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Loads and caches all three models and their configuration JSONs dynamically.
    No dimensions or vocabularies are hardcoded.
    """
    models_registry = {}

    def clean_sd(state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_sd[new_key] = v
        return new_sd

    # 1. Flickr8k
    cfg8_path = os.path.join(MODELS_DIR, "flickr8k_model_config.json")
    pth8_path = os.path.join(MODELS_DIR, "flickr8k_decoder_best.pth")
    with open(cfg8_path, "r", encoding="utf-8") as f:
        cfg8 = json.load(f)
    
    dec8 = PlainLSTMDecoder(
        feature_dim=cfg8.get("feature_dim", cfg8.get("encoder_dim", 2048)),
        embed_size=cfg8["embed_size"],
        hidden_size=cfg8["hidden_size"],
        vocab_size=cfg8["vocab_size"]
    )
    sd8 = torch.load(pth8_path, map_location=DEVICE)
    dec8.load_state_dict(clean_sd(sd8))
    dec8 = dec8.to(DEVICE)
    dec8.eval()
    models_registry["flickr8k"] = {"decoder": dec8, "config": cfg8, "type": "plain"}

    # 2. Flickr30k
    cfg30_path = os.path.join(MODELS_DIR, "flickr30k_attn_model_config.json")
    pth30_path = os.path.join(MODELS_DIR, "flickr30k_attn_decoder_best.pth")
    with open(cfg30_path, "r", encoding="utf-8") as f:
        cfg30 = json.load(f)
    
    dec30 = AttentionLSTMDecoder(
        embed_size=cfg30["embed_size"],
        hidden_size=cfg30["hidden_size"],
        attention_dim=cfg30["attention_dim"],
        vocab_size=cfg30["vocab_size"],
        encoder_dim=cfg30.get("encoder_dim", 2048)
    )
    sd30 = torch.load(pth30_path, map_location=DEVICE)
    dec30.load_state_dict(clean_sd(sd30))
    dec30 = dec30.to(DEVICE)
    dec30.eval()
    models_registry["flickr30k"] = {"decoder": dec30, "config": cfg30, "type": "attention"}

    # 3. MS COCO
    cfg_coco_path = os.path.join(MODELS_DIR, "coco_attn_model_config.json")
    pth_coco_path = os.path.join(MODELS_DIR, "coco_attn_decoder_best.pth")
    with open(cfg_coco_path, "r", encoding="utf-8") as f:
        cfg_coco = json.load(f)

    dec_coco = AttentionLSTMDecoder(
        embed_size=cfg_coco["embed_size"],
        hidden_size=cfg_coco["hidden_size"],
        attention_dim=cfg_coco["attention_dim"],
        vocab_size=cfg_coco["vocab_size"],
        encoder_dim=cfg_coco.get("encoder_dim", 2048)
    )
    sd_coco = torch.load(pth_coco_path, map_location=DEVICE)
    dec_coco.load_state_dict(clean_sd(sd_coco))
    dec_coco = dec_coco.to(DEVICE)
    dec_coco.eval()
    models_registry["coco"] = {"decoder": dec_coco, "config": cfg_coco, "type": "attention"}

    return models_registry


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Standardized preprocessing matching training: 224x224, ImageNet normalization."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return transform(pil_image).unsqueeze(0).to(DEVICE)


# =============================================================================
# 5. PREMIUM GLASSMORPHISM THEME & STYLES (DARK + LIGHT)
# =============================================================================

def inject_theme_styles(theme: str = "dark"):
    """Injects high-end glassmorphism design system for Dark and Light modes."""
    if theme == "dark":
        bg_gradient = """
            radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.18) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(168, 85, 247, 0.12) 0%, transparent 50%),
            linear-gradient(135deg, #090d16 0%, #0d1322 50%, #090d16 100%)
        """
        glass_bg = "rgba(17, 24, 39, 0.68)"
        glass_card = "rgba(22, 31, 52, 0.72)"
        glass_border = "rgba(255, 255, 255, 0.08)"
        glass_border_glow = "rgba(99, 102, 241, 0.35)"
        text_primary = "#f9fafb"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        accent_blue = "#38bdf8"
        accent_purple = "#a855f7"
        sidebar_bg = "rgba(13, 19, 34, 0.85)"
        code_bg = "rgba(10, 15, 28, 0.8)"
        stat_card_bg = "rgba(255, 255, 255, 0.03)"
        theme_btn_bg = "rgba(255, 255, 255, 0.06)"
    else:
        bg_gradient = """
            radial-gradient(circle at 15% 20%, rgba(199, 210, 254, 0.5) 0%, transparent 45%),
            radial-gradient(circle at 85% 75%, rgba(186, 230, 253, 0.5) 0%, transparent 50%),
            radial-gradient(circle at 50% 40%, rgba(243, 232, 255, 0.4) 0%, transparent 40%),
            linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #eef2f6 100%)
        """
        glass_bg = "rgba(255, 255, 255, 0.75)"
        glass_card = "rgba(255, 255, 255, 0.85)"
        glass_border = "rgba(255, 255, 255, 0.9)"
        glass_border_glow = "rgba(99, 102, 241, 0.25)"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        text_muted = "#64748b"
        accent_blue = "#0284c7"
        accent_purple = "#7c3aed"
        sidebar_bg = "rgba(255, 255, 255, 0.85)"
        code_bg = "rgba(241, 245, 249, 0.9)"
        stat_card_bg = "rgba(255, 255, 255, 0.6)"
        theme_btn_bg = "rgba(0, 0, 0, 0.05)"

    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: {text_primary};
    }}

    /* Full App Background */
    .stApp {{
        background: {bg_gradient} !important;
        background-attachment: fixed !important;
    }}

    /* Top Header Bar */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid {glass_border} !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.04);
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px !important;
    }}

    /* Glass Panels / Cards */
    .glass-card {{
        background: {glass_card} !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid {glass_border} !important;
        border-radius: 20px !important;
        padding: 1.6rem !important;
        margin-bottom: 1.4rem !important;
        box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.12), inset 0 1px 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
    }}
    .glass-card:hover {{
        border-color: {glass_border_glow} !important;
        box-shadow: 0 16px 40px 0 rgba(99, 102, 241, 0.16) !important;
    }}

    /* Result Caption Card */
    .caption-box {{
        background: {code_bg};
        border-left: 4px solid {accent_purple};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        font-size: 1.22rem;
        font-weight: 600;
        line-height: 1.6;
        color: {text_primary};
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }}

    /* Badge Pills */
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    .badge-f8k {{
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    .badge-f30k {{
        background: rgba(6, 182, 212, 0.15);
        color: #06b6d4;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }}
    .badge-coco {{
        background: rgba(139, 92, 246, 0.15);
        color: #8b5cf6;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }}

    /* Metric Grid */
    .metric-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 1rem;
    }}
    .stat-pill {{
        background: {stat_card_bg};
        border: 1px solid {glass_border};
        border-radius: 12px;
        padding: 8px 12px;
        text-align: center;
    }}
    .stat-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: {accent_blue};
    }}
    .stat-lbl {{
        font-size: 0.72rem;
        color: {text_muted};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Action Buttons Override */
    .stButton > button {{
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.5) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px -4px rgba(99, 102, 241, 0.7) !important;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #4f46e5 100%) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* Hero Typography */
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: {text_secondary};
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }}

    /* Image Preview Container */
    div[data-testid="stImage"] img {{
        border-radius: 16px !important;
        border: 1px solid {glass_border} !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
    }}

    /* Clean Streamlit widgets */
    div[data-baseweb="select"] > div {{
        background-color: {glass_card} !important;
        border-radius: 12px !important;
        border-color: {glass_border} !important;
        color: {text_primary} !important;
    }}
    div[data-testid="stFileUploader"] {{
        background: {glass_card};
        border: 1px dashed {glass_border_glow};
        border-radius: 16px;
        padding: 12px;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =============================================================================
# 6. APPLICATION HEADER & CONTROLS
# =============================================================================

def render_header():
    """Renders the top hero navigation and theme toggle."""
    col_hero, col_toggle = st.columns([0.82, 0.18])
    with col_hero:
        header_html = """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span class="badge-pill badge-coco">Research Showcase</span>
            <span class="badge-pill badge-f30k">Dataset-Scaling Study</span>
            <span class="badge-pill badge-f8k">ResNet-50 Backbone</span>
        </div>
        <div class="hero-title">Multimodal Image Captioning</div>
        <div class="hero-subtitle">
            Empirical investigation into dataset scaling (Flickr8k &rarr; Flickr30k &rarr; MS COCO) 
            and decoder architecture transition from Plain LSTM to Bahdanau Soft Attention.
        </div>
        """
        render_html(header_html)

    with col_toggle:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        current_theme = st.session_state.theme
        btn_label = "☀️ Switch to Light" if current_theme == "dark" else "🌙 Switch to Dark"
        if st.button(btn_label, key="theme_toggle_btn", use_container_width=True):
            st.session_state.theme = "light" if current_theme == "dark" else "dark"
            st.rerun()


# =============================================================================
# 7. SIDEBAR CONTROLS & GALLERY
# =============================================================================

def render_sidebar() -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
    """Renders interactive sidebar with mode selection, gallery, and uploader."""
    with st.sidebar:
        sidebar_header_html = """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #6366f1, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 18px;">
                👁️
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700;">Control Center</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Config & Visual Inputs</p>
            </div>
        </div>
        """
        render_html(sidebar_header_html)

        # Mode Selection
        mode_choice = st.radio(
            "Evaluation Mode:",
            options=["Compare All 3", "Single Model"],
            index=0 if st.session_state.mode == "Compare All 3" else 1,
            horizontal=True
        )
        st.session_state.mode = mode_choice

        selected_model_key = "coco"
        if mode_choice == "Single Model":
            model_display_names = {
                "MS COCO (LSTM + Attention • SOTA)": "coco",
                "Flickr30k (LSTM + Attention)": "flickr30k",
                "Flickr8k (Plain LSTM • Baseline)": "flickr8k"
            }
            selected_display = st.selectbox(
                "Select Model Checkpoint:",
                options=list(model_display_names.keys()),
                index=0
            )
            selected_model_key = model_display_names[selected_display]

        # ---------------------------------------------------------------------
        # ROOT CAUSE FIX & DIAGNOSIS:
        # 1. Previously, beam_width had min_value=2, completely preventing selection of k=1 (pure greedy).
        # 2. In Compare All 3 mode, beam_width=5 was hardcoded, completely ignoring any slider.
        # 3. In Single Model mode, the slider was only shown for attention models, but hid when switching modes.
        # Now:
        # - The Beam Width slider supports min_value=1 (true greedy) up to max_value=10.
        # - The selected beam_width is universally passed into Flickr30k & MS COCO in BOTH
        #   Single Model mode AND Compare All 3 mode.
        # - When k=1, Attention models operate in true greedy decoding (top-1).
        # ---------------------------------------------------------------------
        show_beam_slider = (mode_choice == "Compare All 3") or (selected_model_key in ("flickr30k", "coco"))
        
        if show_beam_slider:
            st.markdown("#### ⚙️ Decoding Parameters")
            current_k_val = int(st.session_state.get("beam_width", 5))
            beam_width = st.slider(
                "Beam Width (k) for Attention Models:",
                min_value=1,
                max_value=10,
                value=current_k_val,
                step=1,
                key="beam_width_slider",
                help="k=1 is pure greedy decoding (top-1 argmax). Higher k (2-10) expands beam search breadth."
            )
            st.session_state.beam_width = beam_width
            if beam_width == 1:
                st.caption("ℹ️ *k=1 runs exact greedy decoding on the attention models.*")
            else:
                st.caption(f"ℹ️ *k={beam_width} runs length-normalized beam search.*")
        else:
            beam_width = 1  # Flickr8k Plain LSTM is always greedy

        st.markdown("---")

        # Visual Input Source: Sample Gallery vs Upload
        st.markdown("#### 🖼️ Image Selection")
        input_source = st.radio("Choose Input Method:", ["Preloaded Samples", "Upload Image"], horizontal=True)

        input_pil_image = None
        sample_filename = ""
        if input_source == "Preloaded Samples":
            sample_options = {
                "Kayaker in Rapids (Flickr8k / 30k style)": "kayak_rapids.jpg",
                "City Street Crossing (MS COCO style)": "city_street.jpg",
                "Dog Playing with Ball (Flickr style)": "dog_park.jpg",
                "Tabby Cat on Window Sill (COCO style)": "cat_window.jpg",
                "Surfer on Ocean Wave (Action style)": "surfer_wave.jpg",
                "Mountain Biker on Alpine Trail (Outdoor style)": "biker_mountain.jpg"
            }
            sample_label = st.selectbox("Select Benchmark Sample:", list(sample_options.keys()))
            sample_filename = sample_options[sample_label]
            sample_path = os.path.join(SAMPLE_DIR, sample_filename)
            
            if os.path.exists(sample_path):
                input_pil_image = Image.open(sample_path)
            else:
                st.warning("Sample image not found on disk. Please upload an image.")

        else:
            uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    input_pil_image = Image.open(uploaded_file)
                    sample_filename = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading image: {e}")

        # Model Architecture Quick Specs card
        st.markdown("---")
        arch_summary_html = """
        <div style="font-size: 0.8rem; line-height: 1.5; color: #94a3b8;">
            <div style="font-weight: 700; color: #e2e8f0; margin-bottom: 6px;">⚡ Architecture Summary</div>
            • <b>Encoder</b>: ResNet-50 (ImageNet-V2, Frozen)<br>
            • <b>Flickr8k</b>: Pooled (2048,) &rarr; Plain LSTM &rarr; Greedy<br>
            • <b>Flickr30k</b>: Spatial (49, 2048) &rarr; Bahdanau &rarr; Beam Search<br>
            • <b>COCO</b>: Spatial (49, 2048) &rarr; Bahdanau &rarr; Beam Search
        </div>
        """
        render_html(arch_summary_html)

        return mode_choice, input_pil_image, {
            "selected_model_key": selected_model_key,
            "beam_width": beam_width,
            "image_id": sample_filename
        }


# =============================================================================
# 8. CORE INFERENCE EXECUTOR (UNCACED & FRESH)
# =============================================================================

# CRITICAL NOTE ON CACHING:
# This function is intentionally NOT cached with @st.cache_data / @st.cache_resource.
# The user's beam_width argument is forwarded directly into decode_beam_search_attn.
def run_caption_inference(encoder: VisualEncoder,
                          models_registry: Dict[str, Any],
                          model_key: str,
                          pil_image: Image.Image,
                          beam_width: int = 5) -> Tuple[str, float]:
    """
    Runs the exact inference pipeline for a given model:
      - Flickr8k: pooled ResNet-50 feature -> Plain LSTM -> greedy argmax decoding
      - Flickr30k & MS COCO: spatial (49, 2048) grid -> Bahdanau LSTMCell -> length-normalized beam search with beam_width k
    """
    start_time = time.time()
    img_tensor = preprocess_image(pil_image)
    model_entry = models_registry[model_key]
    decoder = model_entry["decoder"]
    cfg = model_entry["config"]

    if model_entry["type"] == "plain":
        # Flickr8k: pooled 2048-dim feature vector (pure greedy argmax)
        pooled_feat = encoder.extract_pooled(img_tensor)
        caption = decode_greedy_plain(
            pooled_feat,
            decoder,
            cfg["word2idx"],
            cfg["idx2word"],
            max_len=cfg["max_len"]
        )
    else:
        # Flickr30k & MS COCO: 49 spatial regions of 2048-dim with Bahdanau attention
        # beam_width is explicitly passed from the caller with no hardcoding
        spatial_feat = encoder.extract_spatial(img_tensor)
        caption = decode_beam_search_attn(
            spatial_feat,
            decoder,
            cfg["word2idx"],
            cfg["idx2word"],
            max_len=cfg["max_len"],
            beam_width=beam_width
        )

    latency = time.time() - start_time
    return caption, latency


# =============================================================================
# 9. RESULT CARD RENDERING COMPONENTS
# =============================================================================

def render_caption_card(model_key: str, caption: str, latency: float, beam_width: int = 5):
    """
    Renders a single frosted glass card for a model's caption with benchmark statistics.
    Uses render_html to prevent any markdown-indented code block leaking.
    Displays the exact beam_width (k) that generated this caption.
    """
    meta = BENCHMARKS[model_key]
    badge_class = "badge-f8k" if model_key == "flickr8k" else ("badge-f30k" if model_key == "flickr30k" else "badge-coco")
    
    if model_key == "flickr8k":
        strategy_label = meta["decoding"]
    elif beam_width == 1:
        strategy_label = "Beam Search (k=1 • Greedy)"
    else:
        strategy_label = f"Beam Search (k={beam_width})"
    
    # Escape caption text to avoid HTML breakage
    clean_caption = html.escape(caption.strip()) if caption else "No caption generated"

    html_content = f"""
    <div class="glass-card" style="border-top: 3px solid {meta['accent']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span class="badge-pill {badge_class}">{meta['dataset_name']}</span>
            <span style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                ⏱️ {latency:.2f}s
            </span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px;">
            <b>{meta['arch_label']}</b> • {strategy_label}
        </div>
        <div class="caption-box" style="border-left-color: {meta['accent']};">
            "{clean_caption}"
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-top: 14px;">
            📊 Test-Set Benchmark Reference
        </div>
        <div class="metric-grid">
            <div class="stat-pill">
                <div class="stat-val" style="color: {meta['accent']};">{meta['bleu4']}</div>
                <div class="stat-lbl">BLEU-4 Score</div>
            </div>
            <div class="stat-pill">
                <div class="stat-val" style="color: {meta['accent']};">{meta['cider']}</div>
                <div class="stat-lbl">CIDEr Score</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 0.72rem; color: #64748b; text-align: right;">
            Training size: {meta['dataset_size']}
        </div>
    </div>
    """
    render_html(html_content)


# =============================================================================
# 10. MAIN APP CONTROLLER
# =============================================================================

def main():
    # Inject active theme stylesheet
    inject_theme_styles(st.session_state.theme)

    # Render Hero Navigation
    render_header()

    # Render Sidebar & Get Inputs
    mode, input_image, opts = render_sidebar()

    # Preload models and encoder
    with st.spinner("Initializing ResNet-50 visual encoder & loading checkpoints..."):
        encoder = load_visual_encoder()
        models_registry = load_all_models()

    # Main Layout Grid: Image Preview & Generation Action
    if input_image is None:
        st.info("👈 Please select a preloaded sample image or upload a photograph in the sidebar to begin.")
        return

    # Image Preview & Action Panel
    col_img, col_act = st.columns([0.45, 0.55])
    with col_img:
        st.image(input_image, caption="Visual Input Query", use_container_width=True)

    # Detect if user changed parameters (beam_width, model, image, or mode)
    current_run_signature = (
        mode,
        opts["selected_model_key"],
        opts["beam_width"],
        opts["image_id"]
    )
    params_changed = (
        st.session_state.last_run_signature is not None
        and st.session_state.last_run_signature != current_run_signature
    )

    with col_act:
        action_headline = '⚡ Multi-Model Scaling Comparison' if mode == 'Compare All 3' else '✨ Single Model Caption Generation'
        
        if mode == 'Compare All 3':
            action_desc = (
                f"Execute inference across all three checkpoints simultaneously on this image. "
                f"Attention decoders (Flickr30k & MS COCO) will use **Beam Width k={opts['beam_width']}**."
            )
        else:
            m_name = BENCHMARKS[opts['selected_model_key']]['dataset_name']
            k_info = f" with **Beam Width k={opts['beam_width']}**" if opts['selected_model_key'] in ('flickr30k', 'coco') else " (Greedy decoding)"
            action_desc = f"Generate caption using the {m_name} checkpoint{k_info}."

        action_panel_html = f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 1.25rem; font-weight: 700;">
                {action_headline}
            </h3>
            <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                {action_desc}
            </p>
        </div>
        """
        render_html(action_panel_html)

        if params_changed and st.session_state.last_results:
            st.info(f"💡 Settings changed (Beam Width k={opts['beam_width']}). Click below to generate fresh captions.")

        if mode == "Compare All 3":
            action_label = f"🚀 Run Comparison Across All 3 Models (k={opts['beam_width']})"
        else:
            action_label = f"⚡ Generate Caption (k={opts['beam_width']})" if opts['selected_model_key'] in ('flickr30k', 'coco') else "⚡ Generate Caption"
            
        generate_clicked = st.button(action_label, key="main_action_btn", use_container_width=True)

    # Execution Flow
    if generate_clicked:
        active_k = opts["beam_width"]
        if mode == "Compare All 3":
            with st.spinner(f"Extracting ResNet-50 visual representations & computing beam search (k={active_k})..."):
                results = {}
                # 1. Flickr8k: Always greedy plain LSTM
                cap8, lat8 = run_caption_inference(encoder, models_registry, "flickr8k", input_image, beam_width=1)
                results["flickr8k"] = {"caption": cap8, "latency": lat8, "k": 1}
                
                # 2. Flickr30k: Uses user-selected beam_width from slider
                cap30, lat30 = run_caption_inference(encoder, models_registry, "flickr30k", input_image, beam_width=active_k)
                results["flickr30k"] = {"caption": cap30, "latency": lat30, "k": active_k}
                
                # 3. MS COCO: Uses user-selected beam_width from slider
                cap_c, lat_c = run_caption_inference(encoder, models_registry, "coco", input_image, beam_width=active_k)
                results["coco"] = {"caption": cap_c, "latency": lat_c, "k": active_k}

                st.session_state.last_results = results
                st.session_state.last_run_signature = current_run_signature

        else:
            target_model = opts["selected_model_key"]
            with st.spinner(f"Generating caption using {BENCHMARKS[target_model]['dataset_name']} (k={active_k})..."):
                cap, lat = run_caption_inference(
                    encoder,
                    models_registry,
                    target_model,
                    input_image,
                    beam_width=active_k
                )
                st.session_state.last_results = {
                    target_model: {
                        "caption": cap,
                        "latency": lat,
                        "k": active_k
                    }
                }
                st.session_state.last_run_signature = current_run_signature

    # Render Results If Available
    if st.session_state.last_results:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🎯 Model Outputs & Comparative Evaluation")

        if mode == "Compare All 3":
            cols = st.columns(3)
            model_keys = ["flickr8k", "flickr30k", "coco"]
            for idx, key in enumerate(model_keys):
                with cols[idx]:
                    if key in st.session_state.last_results:
                        res = st.session_state.last_results[key]
                        render_caption_card(key, res["caption"], res["latency"], res["k"])

            # Scaling Study Qualitative Insights
            observations_html = """
            <div class="glass-card" style="margin-top: 1rem;">
                <h4 style="margin: 0 0 10px 0; font-size: 1.05rem; font-weight: 700; color: #818cf8;">
                    📈 Scaling Study Key Observations
                </h4>
                <div style="font-size: 0.88rem; line-height: 1.6; color: #94a3b8;">
                    • <b>Flickr8k (8k baseline)</b>: Constrained to high-frequency templates and simple subject-action verbs. Lacks localized spatial grounding due to single pooled vector representation.<br>
                    • <b>Flickr30k (31k attention)</b>: Bahdanau attention over 7x7 spatial regions produces richer visual grounding (distinguishing foreground actors vs background terrain).<br>
                    • <b>MS COCO (70k+ attention)</b>: Highest vocabulary diversity and syntactic fluency (+63% BLEU-4, +101% CIDEr improvement over 8k). Demonstrates strong compositionality and multi-object relationships.
                </div>
            </div>
            """
            render_html(observations_html)

        else:
            # Single Model Card Display
            mkey = opts["selected_model_key"]
            if mkey in st.session_state.last_results:
                res = st.session_state.last_results[mkey]
                col_centered = st.columns([0.15, 0.7, 0.15])[1]
                with col_centered:
                    render_caption_card(mkey, res["caption"], res["latency"], res["k"])

    # Bottom Architectural / Benchmark Reference Section
    with st.expander("📚 Dataset-Scaling Research Benchmark Data & Model Architecture Specs", expanded=False):
        st.markdown(
            """
            | Model & Dataset | Training Split Size | Encoder Output | Decoder Architecture | Decoding Strategy | Test BLEU-4 | Test CIDEr |
            |:---|:---|:---|:---|:---|:---|:---|
            | **Flickr8k** | 8,091 images (~40k captions) | Pooled `(2048,)` | Plain LSTM (embed: 256, hidden: 512) | Greedy (Argmax) | **0.1819** | **0.4457** |
            | **Flickr30k** | 31,783 images (~158k captions) | Spatial `(49, 2048)` | LSTMCell + Bahdanau Attn (hidden: 512, att: 256) | Beam Search (k=1..10) | **0.2226** | **0.4737** |
            | **MS COCO** | 70,000+ images (~350k captions) | Spatial `(49, 2048)` | LSTMCell + Bahdanau Attn (hidden: 768, att: 256) | Beam Search (k=1..10) | **0.2974** | **0.9512** |
            
            *Note: All BLEU-4 and CIDEr scores are computed on the respective held-out test splits as part of the scaling study.*
            """
        )


if __name__ == "__main__":
    main()
