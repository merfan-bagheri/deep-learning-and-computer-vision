# 🤖 HW4: Vision Transformers, Recurrent Sequence Modeling & Generative Pre-training (GPT-2 / BERT)

**Author**: Muhammaderfan Bagherinejad

A comprehensive exploration of modern deep learning architectures spanning vision transformers with data augmentation, recurrent neural language models (LSTM), and transformer-based decoder and encoder representations (GPT-2 and BERT).

---

## 📌 Module Breakdown

```mermaid
graph LR
    A[HW4 Modules] --> B[Q1: ViT & Data Augmentation]
    A --> C[Q2: LSTM Language Modeling]
    A --> D[Q3: GPT-2 & BERT Semantic Classification]
```

### 🖼️ Q1: Vision Transformers & Data Augmentation
- **Notebook**: `q1/HW4_Q1_ViT_Data_Augmentation.ipynb`
- **Topics**: Fine-tuning Vision Transformers with advanced data augmentation strategies. Includes preprocessed Hugging Face Arrow datasets and PyTorch model checkpoints (`best_model.pth`, `best_model_with_augmentation.pth`).

### 📜 Q2: LSTM Language Modeling
- **Notebook**: `q2/HW4_Q2_LSTM_Language_Modeling.ipynb`
- **Detailed Docs**: [`q2/README.md`](q2/README.md)
- **Topics**: Multi-layer LSTM language modeling on the WikiText-2 dataset with tied embedding weights, dropout regularization, and perplexity evaluation.

### 💬 Q3: GPT-2 & BERT for Sentiment Classification
- **Notebook**: `q3/HW4_Q3_GPT2_Sentiment_Classification.ipynb`
- **Detailed Docs**: [`q3/README.md`](q3/README.md)
- **Topics**: Architectural variations on decoder-only GPT-2 (Linear Head, Aggregation Layer, Multi-head Self-Attention, Left-to-Right and Right-to-Left Attention) benchmarked against fine-tuned BERT on the Stanford Sentiment Treebank (SST-2).
