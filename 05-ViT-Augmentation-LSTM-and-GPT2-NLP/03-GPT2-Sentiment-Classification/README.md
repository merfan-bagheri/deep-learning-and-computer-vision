# Semantic Classification using SST-2 dataset with Decoder-Only models like GPT-2
**Author**: Mohammad Baqeri
## Introduction

In this assignment, we explore and compare various architectures based on the GPT-2 model for text classification tasks, specifically semantic classification using the SST-2 dataset. The goal is to analyze the performance of different configurations and enhancements made to the base model, focusing on training and validation accuracy, loss metrics, and confusion matrices.

## Goals

- To evaluate the performance of multiple model architectures based on GPT-2.
- To understand how architectural changes affect classification accuracy.
- To identify the best-performing model architecture for the given task.

## Model Architectures

### Model V1: GPT2_clss_v1

#### Architecture
This model uses the base GPT-2 architecture without the language modeling head. It includes a linear classification layer on top to predict the classes.

```python
class GPT2_clss_v1(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(GPT2_clss_v1, self).__init__()
        self.gpt2 = base_model  # Pretrained GPT-2 without the language modeling head
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
```

### Model V2: GPT2_clss_v2

#### Architecture
This model builds on V1 by adding an additional linear aggregation layer to enhance feature extraction before classification. 

```python
class GPT2_clss_v2(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(GPT2_clss_v2, self).__init__()
        self.gpt2 = base_model
        self.aggregate_layer = nn.Linear(self.gpt2.config.hidden_size, self.gpt2.config.hidden_size)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        aggregated_embedding = torch.relu(self.aggregate_layer(cls_embedding))
        logits = self.classifier(aggregated_embedding)
        return logits
```

### Model V3: GPT2_clss_v3

#### Architecture
This model introduces a multi-head self-attention layer to capture complex relationships in the data. It performs attention on the last hidden states from the GPT-2 outputs.

```python
class GPT2_clss_v3(nn.Module):
    def __init__(self, base_model, num_classes=2, num_attention_heads=12):
        super(GPT2_clss_v3, self).__init__()
        self.gpt2 = base_model
        self.attention = nn.MultiheadAttention(embed_dim=self.gpt2.config.hidden_size, num_heads=num_attention_heads, batch_first=True)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        attention_output, _ = self.attention(last_hidden_state, last_hidden_state, last_hidden_state)
        cls_embedding = attention_output[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
```

### Model V4: GPT2_clss_ltr_attention

#### Architecture
This model employs a left-to-right attention mechanism, allowing the model to focus on past tokens while predicting the current token. This can help in tasks requiring sequential context.

```python
class GPT2_clss_ltr_attention(nn.Module):
    def __init__(self, base_model, num_classes=2, num_heads=12):
        super(GPT2_clss_ltr_attention, self).__init__()
        self.gpt2 = base_model
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.gpt2.config.hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        seq_len = last_hidden_state.size(1)
        ltr_mask = torch.tril(torch.ones(seq_len, seq_len)).to(last_hidden_state.device)
        attention_output, _ = self.attention_layer(last_hidden_state, last_hidden_state, last_hidden_state, attn_mask=ltr_mask)
        cls_embedding = attention_output[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
```

### Model V5: GPT2_clss_rtl_attention

#### Architecture
This model utilizes a right-to-left attention mechanism, which can be beneficial for certain tasks that require focus on future tokens while making predictions.

```python
class GPT2_clss_rtl_attention(nn.Module):
    def __init__(self, base_model, num_classes=2, num_heads=12):
        super(GPT2_clss_rtl_attention, self).__init__()
        self.gpt2 = base_model
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.gpt2.config.hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        seq_len = last_hidden_state.size(1)
        rtl_mask = torch.triu(torch.ones(seq_len, seq_len)).to(last_hidden_state.device)
        attention_output, _ = self.attention_layer(last_hidden_state, last_hidden_state, last_hidden_state, attn_mask=rtl_mask)
        cls_embedding = attention_output[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
```

## One-Shot Inference with Fine-Tuned BERT

In addition to the GPT-2 based models, we will also demonstrate one-shot inference using a fine-tuned BERT model on the SST-2 dataset. BERT has shown remarkable performance on various NLP tasks, including sentiment analysis.

### Architecture

The fine-tuned BERT model typically involves passing the input through a BERT tokenizer, obtaining the embeddings, and then passing them through a classification head.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load fine-tuned BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2')

# Function for one-shot inference
def one_shot_inference(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Example usage
text_sample = "This movie was fantastic!"
prediction = one_shot_inference(text_sample)
print(f"Predicted class: {prediction}")
```

## Conclusion

This assignment provides insights into how different architectural choices impact the performance of text classification tasks using transformer-based models. By systematically comparing these models, we can identify the most effective approaches for specific applications, particularly in the realm of semantic classification using the SST-2 dataset.
```
