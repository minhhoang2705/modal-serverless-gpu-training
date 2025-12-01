"""
GPT (Generative Pre-trained Transformer) implementation.

This module implements the core GPT architecture following the approach
from "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Architecture:
    Input → Token Embeddings + Positional Embeddings
         → Transformer Blocks (×N)
         → Layer Norm
         → Output Projection
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration for GPT model architecture."""

    vocab_size: int = 50257  # GPT-2 vocabulary size
    context_length: int = 1024  # Maximum sequence length
    n_layers: int = 12  # Number of transformer blocks
    n_heads: int = 12  # Number of attention heads
    d_model: int = 768  # Embedding dimension
    d_ff: int = 3072  # Feed-forward dimension (typically 4 * d_model)
    dropout: float = 0.1  # Dropout probability

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This is the core component that allows the model to attend to different
    positions in the sequence.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.d_head = config.d_model // config.n_heads

        # Combined linear projection for Q, K, V
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.config.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # scores = (Q @ K^T) / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, d_head]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear transformations with GELU activation in between.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation (smoother than ReLU)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block: Attention + Feed-Forward with residual connections.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Pre-norm architecture (LayerNorm before attention/FFN)
        # Attention block with residual
        x = x + self.dropout(self.attention(self.ln1(x)))

        # Feed-forward block with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) language model.

    This is a decoder-only transformer that can generate text autoregressively.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (share embeddings with output projection)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: [batch_size, seq_len] - input token indices
            targets: [batch_size, seq_len] - target token indices (optional)

        Returns:
            logits: [batch_size, seq_len, vocab_size] - output logits
            loss: scalar tensor (if targets provided)
        """
        batch_size, seq_len = idx.shape

        # Token and position embeddings
        token_emb = self.token_embedding(idx)  # [batch, seq, d_model]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # [seq, d_model]

        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq, vocab_size]

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            idx: [batch_size, seq_len] - conditioning sequence
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top-k tokens

        Returns:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.context_length else \
                       idx[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    config = GPTConfig(
        vocab_size=50257,
        context_length=256,
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
    )

    model = GPT(config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, idx)
    print(f"Input shape: {idx.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
