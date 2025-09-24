# Implementation copied/inspired from: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html


import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    # TODO: Q: Does positional encoding make sense when our protein seqs have already been clipped?

    def __init__(self, d_model, max_len):
        """
        Args:
            d_model: Dimensionality of the input embedding
            max_len: Max len of input sequence, equal to clip_len in Dataloader
        """
        super().__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape (1, max_len, d_model)
        # buffer is for saving parameters that define state of model which
        # are NOT weights/biases (ie no gradient update)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            output: [batch_size, sequence_length, input_dim]
        """
        output = x + self.pe[: x.size(0), :]
        return output


class TransformerEncoderLayer_with_Attn(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout):
        """
        Args:
            input_dim: Dimensionality of the input embedding
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        # TODO: Add parameter to tune MultiheadAttention dropout. Note this is DIFFERENT from linear_net dropout
        self.self_attn = nn.MultiheadAttention(
            input_dim, num_heads, dropout=0.2, batch_first=True
        )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, masks, return_attn=False):
        """
        Args:
            embeddings: [batch_size, sequence_length, input_dim]
            masks: [batch_size, sequence_length], zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size, sequence_length, input_dim]
            attn_weights: [batch_size, sequence_length, sequence_length]
        """

        # Attention part
        if masks is not None:
            key_padding_mask = masks == 0  # flip and convert to boolean
        attn_out, attn_weights = self.self_attn(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )  # returns None for attn_weights if need_weights=False
        output = embeddings + self.dropout(attn_out)
        output = self.norm1(output)

        # MLP part
        linear_out = self.linear_net(output)
        output = output + self.dropout(linear_out)
        output = self.norm2(output)

        return output, attn_weights


class TransformerEncoder_w_Attn(nn.Module):
    def __init__(
        self, num_layers, input_dim, num_heads, dim_feedforward, dropout, max_len
    ):
        super().__init__()

        # Positional encoding to encode sequence positions
        self.positional_encoding = PositionalEncoding(input_dim, max_len)

        # Stack of transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer_with_Attn(
                    input_dim, num_heads, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: [batch_size, sequence_length, input_dim]
            masks: [batch_size, sequence_length], zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size, seq_len, input_dim]
            attention: [batch_size, num_layers, sequence_length, sequence_length] or None
        """

        # Apply positional encoding
        embeddings = self.positional_encoding(embeddings)

        attention = []
        for layer in self.layers:
            embeddings, attn_weights = layer(embeddings, masks, return_attn=True)
            attention.append(attn_weights)
        attention = (
            torch.stack(attention).permute(1, 0, 2, 3)
            if attention[0] is not None
            else None
        )  # [batch_size, num_layers, sequence_length, sequence_length]
        return embeddings, attention
