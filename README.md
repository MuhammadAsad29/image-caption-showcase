<div align="center">

# 🖼️ Multimodal Image Captioning: An Empirical Dataset-Scaling Study

### *Benchmarking ResNet-50 Encoders, Bahdanau Spatial Attention, and Beam Search across Flickr8k, Flickr30k, and MS COCO (70k Subset)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Git LFS](https://img.shields.io/badge/Git-LFS-F05032?style=for-the-badge&logo=git-lfs&logoColor=white)](https://git-lfs.github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[**Explore Architecture**](#-model-architectures--specifications) • [**Empirical Study**](#-dataset-scaling-study--empirical-findings) • [**Showcase App**](#-interactive-streamlit-showcase) • [**Quickstart**](#-quickstart--local-installation) • [**Training Notebooks**](#-training-notebooks)

---

</div>

## 📌 Executive Summary

This research project presents an **empirical investigation into dataset scaling and architectural evolution in vision-language models** for automated image caption generation.

By systematically training and evaluating three deep learning models across progressively larger datasets:
- **Flickr8k**: 8,091 images (~40k captions)
- **Flickr30k**: 31,783 images (~158k captions)
- **MS COCO 2017 (70k Subset)**: 70,000 images (~350k captions sampled from COCO 2017)

We examine how training volume, vocabulary coverage, and spatial attention mechanisms collectively dictate:
1. **Lexical Diversity & Precision**: Moving from high-frequency generic descriptors to fine-grained contextual nomenclature.
2. **Visual Grounding**: How spatial feature grids combined with **Bahdanau Additive Attention** overcome the information bottleneck of global feature pooling.
3. **Inference Dynamics**: The empirical impact of **Length-Normalized Beam Search** ($k=1$ to $10$) on syntactic fluency and repetition penalties.

Accompanying the research is an interactive, dark/light-mode **Streamlit Showcase Application** featuring single-model deep dives, side-by-side comparative benchmarking, preset stress-test images, and custom image inference.

---

## 🔬 Model Architectures & Specifications

All three models leverage a frozen, ImageNet-pretrained **ResNet-50** (`ResNet50_Weights.IMAGENET1K_V2`) as their visual encoder backbone, but implement two fundamentally distinct decoding paradigms:

```
[Input Image (224x224)]
          │
  ┌───────┴───────┐
  │   ResNet-50   │ (Pretrained ImageNet Backbone, Frozen)
  └───────┬───────┘
          │
    ┌─────┴────────────────────────┐
    ▼                              ▼
Global Avg Pooling             conv5_x Spatial Map
 (1 x 2048 vector)              (7 x 7 x 2048 = 49 x 2048)
    │                              │
    ▼                              ▼
Plain LSTM Decoder           Bahdanau Spatial Attention
(Flickr8k Model)                   │
                                   ▼
                             LSTMCell Decoder
                         (Flickr30k & COCO Models)
                                   │
                                   ▼
                       Length-Normalized Beam Search (k=1..10)
```

### Comprehensive Model Matrix

| Specification | Flickr8k Model | Flickr30k Model | MS COCO (70k Subset) |
| :--- | :---: | :---: | :---: |
| **Dataset Source** | Flickr8k | Flickr30k | MS COCO 2017 (`captions_train2017.json`) |
| **Dataset Scale** | 8,091 images (~40k captions) | 31,783 images (~158k captions) | **70,000 images** (350,189 captions) |
| **Split Strategy** | Standard split | Standard split | 80% Train (56k) / 10% Val (7k) / 10% Test (7k) |
| **Vocabulary Size** | 2,988 tokens | 7,689 tokens | 8,237 tokens (`min_freq=5`) |
| **Visual Encoder** | ResNet-50 (Global Pool) | ResNet-50 (`conv5_x` grid) | ResNet-50 (`conv5_x` grid) |
| **Extracted Feature** | `(2048,)` 1D vector | `(49, 2048)` Spatial grid | `(49, 2048)` Spatial grid |
| **Attention Mechanism** | *None (Global bottleneck)* | Bahdanau Additive (`dim=256`) | Bahdanau Additive (`dim=256`) |
| **Embedding Dimension** | 256 | 256 | 256 |
| **Decoder Hidden Size** | 512 (`nn.LSTM`) | 512 (`nn.LSTMCell`) | 768 (`nn.LSTMCell`) |
| **Max Sequence Length** | 37 | 24 | 18 (97th percentile) |
| **Decoding Strategy** | Greedy (Argmax) | Length-Normalized Beam Search | Length-Normalized Beam Search |
| **BLEU-4 Score** | 0.1819 | 0.2226 | **0.2974** |
| **CIDEr Score** | 0.4457 | 0.4737 | **0.9512** |
| **Checkpoint Size** | ~17.6 MB | ~57.8 MB | ~87.0 MB |

---

## 📊 Dataset Scaling Study & Empirical Findings

```
DATASET VOLUME & VOCABULARY PROGRESSION
Flickr8k   (8k imgs)    ──► Vocab: 2,988  ──► Formulaic descriptors ("a dog runs through the grass")
Flickr30k  (31.7k imgs) ──► Vocab: 7,689  ──► Human action & scene context ("a person in blue kayak paddles")
COCO-70k   (70k imgs)   ──► Vocab: 8,237  ──► Complex relational reasoning & prepositional precision
```

### 1. The Global Pooling Bottleneck (Flickr8k)
- **Mechanism**: The Flickr8k decoder receives image information strictly at $t=0$ via a linear projection of the globally pooled 2048-dimensional ResNet-50 vector.
- **Limitation**: Compressing an entire scene into a single vector causes severe information loss in cluttered, multi-object images. The model defaults to dominant statistical priors (e.g., classifying almost any grassy terrain with movement as `"a dog running through the grass"`).

### 2. The Spatial Attention Revolution (Flickr30k)
- **Mechanism**: The decoder queries a $7 \times 7$ feature grid ($49$ visual regions) at every recurrent time step $t$. Bahdanau additive attention dynamically re-weights image regions conditioned on the previous hidden state $h_{t-1}$.
- **Empirical Benefit**: Dramatically improves visual grounding for human attire, gestures, and relative positions. Hallucinated attributes drop significantly.

### 3. Contextual Density & Scaling Dynamics (MS COCO 70k Subset)
- **Design Decision**: Rather than training on the full 118k COCO dataset, a randomized, reproducible **70,000-image subset** (`SUBSET_SIZE = 70000`, seed 42) was sampled from COCO 2017, generating **350,189 captions** partitioned into:
  - **Train**: 280,148 captions (56,000 images)
  - **Validation**: 35,018 captions (7,000 images)
  - **Test**: 35,023 captions (7,000 images)
- **Architecture Scaling**: To accommodate the wider semantic distribution of COCO, the decoder hidden size was expanded from **512 to 768**, combined with a **0.6 dropout** rate and mixed-precision (`torch.cuda.amp`) multi-GPU training.
- **Empirical Benefit**: Yields a dramatic leap to **0.2974 BLEU-4** and **0.9512 CIDEr**. Recognizes complex indoor relationships (appliances, food plating, room architecture) and subtle outdoor interactions that smaller datasets miss entirely.

### 4. Length-Normalized Beam Search Dynamics
Greedy decoding often falls into local probabilistic traps. To combat this, our attention decoders implement beam search tracking the top-$k$ hypotheses with **exponential length penalty normalization**:

$$\text{Score}(Y) = \frac{\sum_{t=1}^T \log P(y_t \mid y_{<t}, X)}{T^\alpha} \quad (\alpha = 0.7)$$

This prevents beam search from artificially penalizing informative, longer descriptions in favor of unnaturally short sentences.

---

## 🚀 Interactive Streamlit Showcase

The repository includes a web application designed with modern aesthetics and dark/light mode support:

```
┌────────────────────────────────────────────────────────────────────────┐
│  MULTIMODAL CAPTIONING — DATASET SCALING STUDY                        │
│  [Dark / Light Mode Toggle]                      [Project GitHub]      │
├────────────────────────────────┬───────────────────────────────────────┤
│ ⚙️ CONTROLS & BENCHMARK SUITE   │ 🖼️ COMPARATIVE MODEL EVALUATION       │
│                                │                                       │
│ • Mode: Compare All 3 Models   │ ┌───────────────┬───────────────────┐ │
│ • Mode: Single Model Deep Dive │ │ Benchmark     │ Generated Caption │ │
│                                │ │ Image Preview │                   │ │
│ • Presets:                     │ └───────────────┴───────────────────┘ │
│   - Kayak Rapids               │                                       │
│   - City Street Crossing       │ 🟢 Flickr8k (ResNet-50 + Plain LSTM)  │
│   - Golden Retriever Park      │    "a man is riding a boat"           │
│   - Domestic Cat Window        │                                       │
│   - Ocean Surfer Wave          │ 🔵 Flickr30k (ResNet-50 + Attention)  │
│   - Mountain Biker Trail       │    "a person in yellow kayak paddling"│
│                                │                                       │
│ • Beam Width Slider: k=1 .. 10 │ 🟣 COCO-70k (ResNet-50 + Attention)  │
│ • Custom Image Upload (.jpg)   │    "a man navigating river rapids"    │
└────────────────────────────────┴───────────────────────────────────────┘
```

### Key UI Features
- **Side-by-Side Comparison**: Run all 3 models simultaneously on the same image to immediately observe lexical and architectural shifts.
- **Beam Width Interactive Sweeping**: Adjust beam width $k \in [1, 10]$ in real-time ($k=1$ activates pure greedy decoding; $k > 1$ runs multi-hypothesis search).
- **Preset Test Suite**: 6 photorealistic benchmark images representing diverse challenging scenarios (rapids, street scenes, animals, dynamic sports).
- **Custom Image Uploader**: Drag and drop any `.png`, `.jpg`, or `.jpeg` file for instant inference.

---

## 📁 Repository Layout

```bash
Image-Caption-Generation/
├── .gitattributes                     # Git LFS tracking rules for *.pth model weights
├── app.py                             # Streamlit showcase application
├── requirements.txt                   # Production runtime dependencies
├── README.md                          # Project documentation
│
├── models/                            # Checkpoints & vocabulary configurations
│   ├── flickr8k_decoder_best.pth      # Flickr8k decoder checkpoint (Plain LSTM)
│   ├── flickr8k_model_config.json     # Flickr8k vocabulary (2,988 tokens) & configs
│   ├── flickr30k_attn_decoder_best.pth# Flickr30k decoder checkpoint (Attention)
│   ├── flickr30k_attn_model_config.json# Flickr30k vocabulary (7,689 tokens) & configs
│   ├── coco_attn_decoder_best.pth     # MS COCO 70k decoder checkpoint (Attention)
│   └── coco_attn_model_config.json    # MS COCO vocabulary (8,237 tokens) & configs
│
├── notebooks/                         # End-to-end model training notebooks
│   ├── flickr8k.ipynb                 # ResNet-50 + Plain LSTM training pipeline
│   ├── flickr30k.ipynb                # ResNet-50 + Bahdanau Attention pipeline
│   └── coco70k.ipynb                  # MS COCO 70,000-image subset attention training
│
└── sample_images/                     # Benchmark showcase image suite
    ├── kayak_rapids.jpg               # Water sports & rapid motion
    ├── city_street.jpg                # Dense urban street crossing
    ├── dog_park.jpg                   # Animal action in outdoor terrain
    ├── cat_window.jpg                 # Static indoor domestic scene
    ├── surfer_wave.jpg                # Extreme sports & ocean scenery
    └── biker_mountain.jpg             # Single person outdoor trail cycling
```

---

## 💻 Quickstart & Local Installation

### Prerequisites
- **Python 3.10+**
- **Git & Git LFS** (Git Large File Storage is required to pull the pre-trained weights)

### 1. Clone the Repository with Git LFS
```bash
# Clone the repository
git clone https://github.com/MuhammadAsad29/image-caption-showcase.git
cd image-caption-showcase

# Ensure Git LFS pulls the .pth checkpoint files
git lfs install
git lfs pull
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Showcase Web App
```bash
streamlit run app.py
```
The application will launch automatically at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

Deploying this showcase to **Streamlit Community Cloud** takes under 2 minutes:

1. Fork or push this repository to your GitHub account (`MuhammadAsad29/image-caption-showcase`).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and configure:
   - **Repository**: `MuhammadAsad29/image-caption-showcase`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

> **Note on Git LFS in Streamlit Cloud**: Streamlit Community Cloud natively supports Git LFS up to standard repository size limits. The model checkpoints will be fetched during container initialization.

---

## 📓 Training Notebooks & Methodology

Each model has a corresponding self-contained Jupyter notebook in [`notebooks/`](notebooks/) detailing the entire training workflow:

### [`notebooks/coco70k.ipynb`](notebooks/coco70k.ipynb) — Large-Scale Attention Pipeline
- **Dataset**: Downloads official MS COCO 2017 train images and annotations (`captions_train2017.json`).
- **Sampling Strategy**: Subsets **70,000 unique images** (`random.seed(42)`), compiling **350,189 caption pairs**.
- **Preprocessing & Vocab**: Applies regex normalization (`[^a-z\s]`), wraps with `startseq`/`endseq`, builds an **8,237-token** vocabulary (`min_freq=5`), and clips sequence lengths to the 97th percentile (**18 tokens**).
- **Splits**: Implements an 80/10/10 split on unique images (280,148 train, 35,018 val, 35,023 test).
- **Feature Caching**: Multithreaded image download (`ThreadPoolExecutor`, 64 workers) with batch extraction through ResNet-50 to cached `.npy` spatial feature maps (`(49, 2048)`).
- **Decoder**: Scaled `LSTMCell` with `hidden_size=768`, `attention_dim=256`, and `dropout=0.6`, trained via mixed-precision AMP on 2 GPUs (`DataParallel`).

### [`notebooks/flickr30k.ipynb`](notebooks/flickr30k.ipynb) — Mid-Scale Attention Pipeline
- **Dataset**: 31,783 images, ~158k captions.
- **Model**: ResNet-50 spatial features (`49 x 2048`) with Bahdanau attention (`dim=256`), `LSTMCell` (`hidden=512`), and a **7,689-token** vocabulary.

### [`notebooks/flickr8k.ipynb`](notebooks/flickr8k.ipynb) — Small-Scale Baseline Pipeline
- **Dataset**: 8,091 images, ~40k captions.
- **Model**: ResNet-50 Global Average Pooling (`2048` vector) + Plain `nn.LSTM` (`hidden=512`) with a **2,988-token** vocabulary and greedy argmax inference.

---

## 🛠️ Tech Stack & Credits

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/stable/index.html)
- **Computer Vision Backbone**: ImageNet-pretrained ResNet-50 (He et al., 2015)
- **Attention Architecture**: Bahdanau Additive Attention (Bahdanau et al., 2014; Xu et al., "Show, Attend and Tell", 2015)
- **Web App Framework**: [Streamlit](https://streamlit.io/)
- **Image Processing**: [Pillow](https://python-pillow.org/) & [NumPy](https://numpy.org/)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Developed with ❤️ by **[Muhammad Asad](https://github.com/MuhammadAsad29)**

*If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!*

</div>
