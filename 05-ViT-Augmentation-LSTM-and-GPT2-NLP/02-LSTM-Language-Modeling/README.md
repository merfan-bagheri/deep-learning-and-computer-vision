# LSTM Model for Language Modeling

**Author**: Mohammad Baqeri

## Model Architecture

The LSTM (Long Short-Term Memory) model used for language modeling is defined as follows:

### Model Definition

The model is implemented as a subclass of `nn.Module` in PyTorch. Below is the code for the LSTM model:

```python
import torch
import torch.nn as nn
import math

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        if tie_weights:
            assert embedding_dim == hidden_dim, 'embedding_dim must equal hidden_dim'
            self.embedding.weight = self.fc.weight
            
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)          
        output = self.dropout(output) 
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim, self.hidden_dim).uniform_(-init_range_other, init_range_other) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-init_range_other, init_range_other) 

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell
```

### Parameters

- **vocab_size**: The size of the vocabulary, representing the total number of unique tokens.
- **embedding_dim**: The dimension of the embedding layer, typically set to 1024.
- **hidden_dim**: The dimension of the hidden state in the LSTM, also typically set to 1024.
- **num_layers**: The number of layers in the LSTM, often set to 2.
- **dropout_rate**: The dropout rate for regularization, commonly set to 0.65.
- **tie_weights**: A boolean flag indicating whether to tie the weights between the embedding and the final output layer.

### Model Initialization

The model's weights are initialized in the `init_weights` method, which ensures that:

- The embedding weights are uniformly distributed within a specified range.
- The linear layer weights are also initialized uniformly based on the hidden dimension.
- The biases for the linear layer are initialized to zero.
- The weights for the LSTM cells are initialized for each layer.

### Hidden State Initialization

The `init_hidden` method initializes the hidden state of the LSTM with zeros, which is essential for maintaining memory across sequences.

### Detaching the Hidden State

To prevent backpropagation through the entire history of hidden states, the `detach_hidden` method detaches the hidden state from the computation graph.

### Model Instantiation

The model can be instantiated as follows:

```python
vocab_size = len(vocab)
embedding_dim = 1024            
hidden_dim = 1024                
num_layers = 2                   
dropout_rate = 0.65             
tie_weights = True                  

model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
```

## Loading and Saving the Model

### Saving the Model

After training the model, you can save its state using `torch.save`:

```python
torch.save(model.state_dict(), 'lstm_model.pth')
```

### Loading the Model

To load the model later, you can do the following:

```python
# Create an instance of the model
model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)

# Load the model state
model.load_state_dict(torch.load('lstm_model.pth'))

# Set the model to evaluation mode
model.eval()
```

### Using the Model

Once the model is loaded, you can generate predictions as follows:

```python
# Initialize hidden state
hidden = model.init_hidden(batch_size, device)

# Sample input (token ids)
input_seq = torch.tensor([[...]]).to(device)  # Replace with actual token IDs

# Forward pass
with torch.no_grad():
    predictions, hidden = model(input_seq, hidden)
```

## Conclusion

This LSTM model provides a robust framework for language modeling tasks on datasets like WikiText-2. With its ability to learn long-term dependencies, it can generate coherent text sequences based on the learned patterns in the data.

## References

- PyTorch Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html)
- https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf
- https://lukesalamone.github.io/posts/what-is-temperature/
```
