import torch
import torch.nn as nn
import math

class AudioTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = 4

        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape input tensor to [batch_size, sequence_length, input_size]
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        x = x.permute(0, 2, 1)

        # Apply linear transformation to the input
        embedded = self.embedding(x)  # [batch_size, sequence_length, hidden_size]

        # Positional encoding
        embedded = self.positional_encoding(embedded)

        # Reshape embedded tensor to [sequence_length, batch_size, hidden_size]
        embedded = embedded.permute(1, 0, 2)

        # Transformer encoder forward propagation
        encoded = self.transformer_encoder(embedded)  # [sequence_length, batch_size, hidden_size]

        # Take the last output feature
        last_encoded = encoded[-1, :, :]  # [batch_size, hidden_size]

        # Fully connected layer for classification
        output = self.fc(last_encoded)  # [batch_size, num_classes]

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        self.hidden_size = hidden_size

        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_seq_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
