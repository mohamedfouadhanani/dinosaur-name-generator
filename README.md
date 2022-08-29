# Dinosaur Name Generator

## Introduction

A Recurrent Neural Network (LSTM) based PyTorch model that **generates dinosaur names** with the help of previously seen dinosaur names.

## Neural Network Architecture

```python
class Model(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimensions, hidden_size, n_layers, keep_prob):
        super(Model, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.keep_prob = keep_prob

        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dimensions)
        self.lstm = nn.LSTM(input_size=self.embedding_dimensions, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=self.keep_prob)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocabulary_size)

    def forward(self, X, state_previous):
        h_previous, c_previous = state_previous
        embedding = self.embedding(X)
        y_hat, state_current = self.lstm(embedding, (h_previous, c_previous))
        y_hat = self.dropout(y_hat)
        y_hat = self.fc(y_hat)

        return y_hat, state_current
```
