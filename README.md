# 🧠 MiniHealthLM

A **lightweight and secure Transformer language model** trained from scratch on a single medical textbook—[*Grant's Atlas of Anatomy*](https://uotechnology.edu.iq/dep/bme/english/Pages/Lectures/anatomy%20first%20course/Grant's%20Atlas%20of%20Anatomy.pdf).  
MiniHealthLM is designed for rapid experimentation, low-resource environments, and educational research in healthcare language modeling.

---

## 🎯 Project Purpose

This project explores how to build and pretrain a **domain-specific LLM from scratch** using:

- Only a single PDF textbook as the corpus
- Custom tokenizer trained from that corpus
- Privacy-aware data sanitization
- Lightweight Qwen-style Transformer architecture

It aims to demonstrate how **LLMs can be scaled down** and securely trained for specific domains like **medicine and anatomy**, while remaining fully transparent and modular.

---

## 🏗️ What I Built

MiniHealthLM includes a complete **end-to-end pretraining pipeline**:

- 📖 PDF ➜ plain text (`extract_text.py`)
- 🧼 Secure preprocessing (`sanitize_data.py` using PII filters)
- 🔡 Tokenizer training (Byte-Level BPE)
- 🧠 Transformer pretraining (Qwen-style GQA + RoPE + RMSNorm)
- 💾 Checkpointing & loss tracking with TQDM

Training Dataset:  
📘 **Grant's Atlas of Anatomy**  
➡️ [PDF Link](https://uotechnology.edu.iq/dep/bme/english/Pages/Lectures/anatomy%20first%20course/Grant's%20Atlas%20of%20Anatomy.pdf)

---

## 🧬 Architecture Overview

| Component       | Description                          |
|----------------|--------------------------------------|
| Embedding       | Token Embedding (Byte BPE)          |
| Layers          | 20 Transformer blocks                |
| Hidden Size     | 3072                                 |
| Attention       | Grouped Query Attention (GQA)        |
| Heads           | 16 query heads, 4 key-value heads    |
| Norm            | RMSNorm                              |
| Positional Bias | Rotary Position Embeddings (RoPE)    |
| FFN             | SiLU MLP (4x expansion)              |
| Context Length  | 1024 tokens                          |
| Params (est.)   | ~0.6 Billion                         |

---

## ✨ Novelty

- ⚕️ **Domain-First**: Model trained purely on anatomical medical text
- 🔐 **Secure Pretraining**: Removes PII and sensitive content before tokenization
- 🔬 **Small Yet Complete**: Implements modern LLM features like RoPE, GQA, and RMSNorm
- 🧠 **Trained From Scratch**: No pretraining dependency or transfer — 100% domain-rooted
- ⚡ **Single-GPU Ready**: Efficient enough to run on local or lab hardware

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Text from PDF
```bash
python extract_text.py
```

### 3. Sanitize (Remove PII)
```bash
python sanitize_data.py
```

### 4. Train Tokenizer
```bash
python train_tokenizer.py
```

### 5. Launch Pretraining
```bash
python train.py
```

---

## 🗂️ Folder Structure

```
MiniHealthLM/
├── checkpoints/              # Saved model checkpoints
├── data/
│   ├── corpus.pdf            # Grant's Atlas PDF
│   ├── corpus.txt            # Cleaned training text
│   └── tokenizer/            # Tokenizer files
├── src/
│   ├── config.py             # Model settings
│   ├── model.py              # Qwen-style transformer
│   ├── dataset.py            # DataLoader
│   └── utils.py              # Checkpointing utilities
├── extract_text.py
├── sanitize_data.py
├── train_tokenizer.py
├── test_tokenizer.py
├── train.py
└── requirements.txt
```

---

## 👨‍💻 Author

**Elias Hossain**  
_Machine Learning Researcher | Secure LLMs | Biomedical NLP_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
