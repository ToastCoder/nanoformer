# nanoformer
# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Feed Forward Block Class
class FeedForward(nn.Module):
    """
    This block processes each token independently by expanding its representation
    to a higher-dimensional space and then projecting it back. This allows the
    model to learn complex patterns and transformations for each token based on
    the information it gathered during the attention phase.
    """

    def __init__(self, embedding_dims: int):
        """
        Initializes the feed-forward network.

        Parameters:
            embedding_dims (int): The dimensionality of the token embeddings.
        """

        super().__init__()

        # Sequence of layers that process each token individually.
        self.net = nn.Sequential(
            # Linearly project the input dimension to 4x its size to create "extra space" for learning.
            nn.Linear(embedding_dims, 4 * embedding_dims),
            # Introduce non-linearity so the model can learn complex, non-straightforward relationships.
            nn.ReLU(),
            # Project back down to the original embedding size so it matches the rest of the model.
            nn.Linear(4 * embedding_dims, embedding_dims),
        )

    def forward(self, input_tensor):
        """
        Passes the input through the feed-forward network.

        Parameters:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dims).

        Returns:
            torch.Tensor: The processed tensor of the same shape.
        """

        # Pass the input through our defined network and return the result.
        return self.net(input_tensor)


# Multi Head Attention Class
class MultiHeadAttention(nn.Module):
    """
    This component is the heart of the transformer. It allows tokens in a sequence
    to 'look at' other tokens to build context. By splitting the embedding into
    multiple heads, the model can simultaneously focus on different relationships,
    such as grammar, subject-verb agreement, or long-range dependencies.
    """

    def __init__(self, embedding_dims: int, num_heads: int):
        """
        Initializes the Multi-Head Attention mechanism.

        Parameters:
            embedding_dims (int): The dimensionality of the token embeddings.
            num_heads (int): The number of parallel attention heads to use.
        """

        super().__init__()

        # Check that our dimension splits evenly across the number of heads.
        assert embedding_dims % num_heads == 0, (
            "Embedding dim must be divisible by num_heads"
        )

        self.num_heads = num_heads
        self.head_dims = embedding_dims // num_heads

        # Linear layers to project the input into Query, Key, and Value spaces.
        self.query = nn.Linear(embedding_dims, embedding_dims)
        self.key = nn.Linear(embedding_dims, embedding_dims)
        self.value = nn.Linear(embedding_dims, embedding_dims)

        # Final projection layer to mix the results from all heads back together.
        self.output_projection = nn.Linear(embedding_dims, embedding_dims)

    def forward(self, input_tensor):
        """
        Performs the forward pass for Multi-Head Attention.

        Parameters:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dims).

        Returns:
            torch.Tensor: The output tensor after attending to other tokens.
        """

        # Get the dimensions of our input for reshaping later.
        batch_size, sequence_length, embedding_dims = input_tensor.shape

        # Generate Queries, Keys, and Values, then reshape them to separate the heads.
        # Transposing to ensure the sequence length and head dimensions are in the right order for math.
        query = (
            self.query(input_tensor)
            .view(batch_size, sequence_length, self.num_heads, self.head_dims)
            .transpose(1, 2)
        )
        key = (
            self.key(input_tensor)
            .view(batch_size, sequence_length, self.num_heads, self.head_dims)
            .transpose(1, 2)
        )
        value = (
            self.value(input_tensor)
            .view(batch_size, sequence_length, self.num_heads, self.head_dims)
            .transpose(1, 2)
        )

        # Calculate similarity scores by multiplying Query with Key.
        # Divide by square root of head dimension to keep gradients stable.
        scores = (query @ key.transpose(-2, -1)) / (self.head_dims**0.5)

        # Create a causal mask: a lower triangular matrix of ones.
        mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=input_tensor.device)
        ).view(1, 1, sequence_length, sequence_length)

        # Fill the "future" positions with -infinity so they have zero probability after softmax.
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute probabilities so that every token's attention weights sum to 1.
        attention_weights = F.softmax(scores, dim=-1)

        # Apply weights to the Values to extract context.
        out = attention_weights @ value
        # Move the head dimension back to the end and flatten to restore original tensor shape.
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_dims)
        )

        # Project the final concatenated output.
        return self.output_projection(out)


# Transformer Block Class
class TransformerBlock(nn.Module):
    """
    A transformer block is a self-contained unit that performs a single 'round' of
    processing. It contains an attention mechanism to gather context and a
    feed-forward network to process that context. We use normalization and
    residual connections to ensure that the model remains stable even as the
    information passes through many such blocks.
    """

    def __init__(self, embedding_dims: int, num_heads: int):
        """
        Initializes the Transformer Block.

        Parameters:
            embedding_dims (int): The dimensionality of the token embeddings.
            num_heads (int): The number of attention heads to use.
        """

        super().__init__()

        # Define the building blocks for this layer.
        self.attention = MultiHeadAttention(embedding_dims, num_heads)
        self.feed_forward = FeedForward(embedding_dims)
        self.norm1 = nn.LayerNorm(embedding_dims)
        self.norm2 = nn.LayerNorm(embedding_dims)

    def forward(self, input_tensor):
        """
        Performs the forward pass for the Transformer Block.

        Parameters:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dims).

        Returns:
            torch.Tensor: The output tensor after attention and feed-forward processing.
        """

        # Normalize the input before attention, perform attention, and add it back to the input (residual).
        input_tensor = input_tensor + self.attention(self.norm1(input_tensor))

        # Normalize the updated input before the feed-forward, perform the processing, and add it back.
        input_tensor = input_tensor + self.feed_forward(self.norm2(input_tensor))

        return input_tensor


class NanoFormer(nn.Module):
    """
    The NanoFormer is the complete language model. It manages token and position
    embeddings, propagates the hidden state through a stack of transformer blocks,
    and maps the final refined representations back to the vocabulary to predict
    the next token in the sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dims: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
    ):
        """
        Initializes the complete NanoFormer model.

        Parameters:
            vocab_size (int): The size of the vocabulary.
            embedding_dims (int): The dimensionality of the token embeddings.
            block_size (int): The maximum length of a text segment.
            num_heads (int): The number of attention heads per layer.
            num_layers (int): The number of transformer layers in the stack.
        """

        super().__init__()

        # The look-up table that converts token IDs into vectors.
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dims)

        # The look-up table that stores positional information for tokens.
        self.position_embedding_table = nn.Embedding(block_size, embedding_dims)

        # Create a chain of transformer blocks that will process the data sequentially.
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dims, num_heads) for _ in range(num_layers)]
        )

        # A final normalization layer to clean up the output of the transformer stack.
        self.layer_norm = nn.LayerNorm(embedding_dims)

        # A final linear layer to map the hidden vectors back to the size of the vocabulary.
        self.language_model_head = nn.Linear(embedding_dims, vocab_size)

    def forward(self, input_indices: torch.Tensor):
        """
        Performs the forward pass of the NanoFormer model.

        Parameters:
            input_indices (torch.Tensor): The input tensor of token indices of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The logits representing predicted scores for the next token, of shape (batch_size, sequence_length, vocab_size).
        """

        # Get the batch size and the sequence length from the input.
        batch_size, sequence_length = input_indices.shape

        # Convert token IDs to vectors.
        token_embeddings = self.token_embedding_table(input_indices)

        # Generate position IDs (0, 1, 2...) and convert them to vectors.
        position_indices = torch.arange(sequence_length, device=input_indices.device)
        position_embeddings = self.position_embedding_table(position_indices)

        # Add token and position information together.
        hidden_state = token_embeddings + position_embeddings

        # Propagate the combined state through all transformer blocks.
        hidden_state = self.blocks(hidden_state)

        # Normalize the result.
        hidden_state = self.layer_norm(hidden_state)

        # Project vectors to vocabulary size to get raw scores for each possible next word.
        logits = self.language_model_head(hidden_state)

        return logits
