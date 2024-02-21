import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    def __init__(self, num_categories: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x):
        # print(f"num_categories: {self.num_categories}")
        # print(f"embedding_dim: {self.embedding_dim}")
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(embedding_dim) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.embedding_dim)


#     def __init__(self, features: int, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization(features)

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         return self.norm(x)


class DiscreteInputModule(nn.Module):
    def __init__(self, embedding_layers: nn.ModuleList, offset: int) -> None:
        super().__init__()
        # Define the embedding layer for the categorical feature
        # Offset is the number of continuous variables
        self.embedding_layers = embedding_layers
        self.offset = offset

    def forward(self, x):
        # print(f"offset: {self.offset}")
        # Assuming x is of shape (batch size, T, number_of_vars)
        # Separate the continuous and categorical features
        continuous_features = x[:, :, 0 : self.offset]  # Shape: (batch size, T, offset)
        categorical_features = x[
            :, :, self.offset :
        ]  # Shape: (batch size, T, number_of_vars - offset)

        output = continuous_features
        # Run each categorical feature through their respective embedding layer
        for i, layer in enumerate(self.embedding_layers):
            embedded_feature = layer(
                categorical_features[:, :, i].long()
            )  # Shape: (batch size, T, embedding_dim[i])
            output = torch.cat([output, embedded_feature], dim=-1)

        # embedded_feature = self.embedding_layers(
        #     categorical_feature
        # )  # Shape: (batch size, T, embedding_dim)

        # # Concatenate the continuous feature and the embedded categorical feature
        # output = torch.cat(
        #     [continuous_feature, embedded_feature], dim=-1
        # )  # Shape: (batch size, T, 1+embedding_dim)

        return output

        # class BabyDiscreteInputModule(nn.Module):
        def __init__(self, embedding_layer: InputEmbeddings):
            super().__init__()
            # Define the embedding layer for the categorical feature
            self.embedding_layers = embedding_layer

        def forward(self, x):
            # Assuming x is of shape (batch size, T, 2)
            # Separate the continuous and categorical features
            continuous_feature = x[:, :, 0:1]  # Shape: (batch size, T, 1)
            categorical_feature = x[:, :, 1].long()  # Shape: (batch size, T)

            # Run the categorical feature through the embedding layer
            embedded_feature = self.embedding_layers(
                categorical_feature
            )  # Shape: (batch size, T, embedding_dim)

            # Concatenate the continuous feature and the embedded categorical feature
            output = torch.cat(
                [continuous_feature, embedded_feature], dim=-1
            )  # Shape: (batch size, T, 1+embedding_dim)

            return output


# class ContextualModule(nn.Module):
#     def __init__(self, features: int, layers: nn.ModuleList)


class EncoderInput(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        return self.input_fc(x)  # Output shape: [batch_size, sequence_length, d_model]


# For now this is the same as EncoderInput, but I may need to change
# input_dim to num_predicted_features sometime later
class DecoderInput(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        return self.input_fc(x)  # Output shape: [batch_size, sequence_length, d_model]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(f"x.size(): {x.size()}")
        # print(f"other size: {self.pe[:, : x.shape[1], :].size()}")
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # print(f"d_k: {d_k}")
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # print(f"attention_scores size: {attention_scores.size()}")
        # print(f"mask size: {mask.size()}")
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class LinearMapping(nn.Module):
    # Last part of the time series transformer model
    # This is the time-series equivalent of a projection layer
    def __init__(self, d_model, num_predicted_features) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_predicted_features)

    def forward(self, x) -> None:
        return self.linear(x)


# NOTE: NEED TO CHANGE THIS, VOCAB_SIZE IS IRRELEVANT HERE
# class ProjectionLayer(nn.Module):
#     def __init__(self, d_model, vocab_size) -> None:
#         super().__init__()
#         self.proj = nn.Linear(d_model, vocab_size)

#     def forward(self, x) -> None:
#         # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
#         return self.proj(x)


class Transformer(nn.Module):
    def __init__(
        self,
        # time_embedding: InputEmbeddings,
        src_discrete_input_module: DiscreteInputModule,
        tgt_discrete_input_module: DiscreteInputModule,
        encoder_input: EncoderInput,
        decoder_input: DecoderInput,
        encoder: Encoder,
        decoder: Decoder,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        linear_mapping: LinearMapping,
    ) -> None:
        super().__init__()
        # self.time_embedding = time_embedding
        self.src_discrete_input_module = src_discrete_input_module
        self.tgt_discrete_input_module = tgt_discrete_input_module
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_mapping = linear_mapping

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        # (32, 12, 1)
        # print(f"src.size(): {src.size()}")
        src = self.src_discrete_input_module(src)
        # print(f"src.size() after: {src.size()}")
        src = self.encoder_input(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        # (batch, seq_len, d_model)
        # tgt = self.tgt_embed(tgt)
        # print(f"tgt.size(): {tgt.size()}")
        tgt = self.tgt_discrete_input_module(tgt)
        tgt = self.decoder_input(tgt)
        # print(f"tgt size: {tgt.size()}")
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def map(self, x):
        return self.linear_mapping(x)

    # def project(self, x):
    #     # (batch, seq_len, vocab_size)
    #     return self.projection_layer(x)


def build_transformer(
    input_dim: int,
    discrete_var_dims: list,
    discrete_embedding_dims: list,
    src_seq_len: int,
    tgt_seq_len: int,
    num_predicted_features: int = 1,
    d_model: int = 512,
    N: int = 2,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
    offset: int = 2,
) -> Transformer:
    # Calculate embedding layer input dimensions

    # Create an embedding layer for each discrete variable
    src_embedding_layers = []
    tgt_embedding_layers = []
    for dim_in, dim_out in zip(discrete_var_dims, discrete_embedding_dims):
        # pass
        src_embed = InputEmbeddings(dim_in, dim_out)
        tgt_embed = InputEmbeddings(dim_in, dim_out)

        src_embedding_layers.append(src_embed)
        tgt_embedding_layers.append(tgt_embed)

    # Number of continuous variables
    src_disc = DiscreteInputModule(nn.ModuleList(src_embedding_layers), offset)
    tgt_disc = DiscreteInputModule(nn.ModuleList(tgt_embedding_layers), offset)

    # src_disc = BabyDiscreteInputModule(src_embed)
    # tgt_disc = BabyDiscreteInputModule(tgt_embed)

    encoder_input = EncoderInput(input_dim, d_model)

    # May have to change this -- input_dim vs num_predicted_features
    decoder_input = DecoderInput(input_dim, d_model)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create linear mapping
    linear_mapping = LinearMapping(d_model, num_predicted_features)

    # Create the transformer
    transformer = Transformer(
        src_disc,
        tgt_disc,
        encoder_input,
        decoder_input,
        encoder,
        decoder,
        src_pos,
        tgt_pos,
        linear_mapping,
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
