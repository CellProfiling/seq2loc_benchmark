import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.transformer_w_attn import TransformerEncoder_w_Attn


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: [batch_size, sequence_length, embeddings_dim]
            masks: [batch_size, sequence_length] zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size,embeddings_dim] #[batch_size, embeddings_dim]
        """
        assert torch.is_floating_point(embeddings)
        embeddings = embeddings.masked_fill(masks[:, :, None] == False, float("-inf"))
        emb_max = torch.max(embeddings, dim=1)
        output = emb_max.values
        index = emb_max.indices

        attention = torch.zeros_like(masks)
        attention.scatter_(1, index, 1)

        return output, attention


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: [batch_size, sequence_length, embeddings_dim]
            masks: [batch_size, sequence_length] zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size, embeddings_dim]
        """
        embeddings *= masks[:, :, None]
        lengths = torch.sum(masks, dim=1, keepdim=True)
        output = torch.sum(embeddings, dim=1) / lengths

        attention = masks / lengths

        return output, attention


class LightAttentionPool(nn.Module):
    # Implementation based on: https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    def __init__(
        self,
        input_dim,
        kernel_size=9,
        conv_dropout=0.2,
    ):
        super().__init__()
        self.feature_convolution = nn.Conv1d(
            input_dim, input_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.attention_convolution = nn.Conv1d(
            input_dim, input_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.softmax = nn.Softmax(dim=-1)
        self.conv_dropout = nn.Dropout(conv_dropout)

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: [batch_size, sequence_length, input_dim]
            masks: [batch_size, sequence_length] zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size,input_dim*2]
            attention: [batch_size, input_dim, seq_len], and sum over each embed_dim across seq_len equals 1
        """

        embeddings = embeddings.permute(0, 2, 1)

        output = self.feature_convolution(embeddings)
        output = self.conv_dropout(output)
        attention = self.attention_convolution(embeddings)
        attention = attention.masked_fill(masks[:, None, :] == False, -1e9)

        output1 = torch.sum(output * self.softmax(attention), dim=-1)
        output2, _ = torch.max(output, dim=-1)
        output = torch.cat([output1, output2], dim=-1)

        attention = self.softmax(torch.sum(attention, dim=1))

        return output, attention


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.2):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mean_pool = MeanPool()

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: [batch_size, input_dim, sequence_length]
            masks: [batch_size, sequence_length], zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size, input_dim]
            attn_weights: [batch_size, seq_len, seq_len], and sum over last dim is equal to 1
        """
        if masks is not None:
            key_padding_mask = masks == 0  # flip and convert to boolean

        attn_out, attention = self.self_attn(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )  # returns None for attn_weights if need_weights=False
        output, _ = self.mean_pool(attn_out, masks)

        attention = F.softmax(attention.mean(dim=1), dim=-1)

        return output, attention


class TransformerPool(nn.Module):
    def __init__(self, input_dim, max_len, num_layers=1, num_heads=8, dropout=0.2):
        super().__init__()
        dim_feedforward = input_dim

        self.transformer = TransformerEncoder_w_Attn(
            num_layers,
            input_dim,
            num_heads,
            dim_feedforward,
            dropout,
            max_len,
        )
        self.mean_pool = MeanPool()

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: [batch_size, input_dim, sequence_length]
            masks: [batch_size, sequence_length], zero padding used for the shorter sequences in the batch
        Returns:
            output: [batch_size, input_dim]
            attn_maps: [batch_size, num_layers, sequence_length, sequence_length]
        """

        transformer_out, attention = self.transformer(embeddings, masks)
        output, _ = self.mean_pool(transformer_out, masks)

        attention = F.softmax(attention.mean(dim=1), dim=-1)
        
        return output, attention
