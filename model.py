import math
import torch
import torch.nn as nn

from vocabulary import PAD_IDX


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)].detach())


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1,  keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, dff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h       = h
        self.d_k     = d_model // h
        self.W_q     = nn.Linear(d_model, d_model, bias=False)
        self.W_k     = nn.Linear(d_model, d_model, bias=False)
        self.W_v     = nn.Linear(d_model, d_model, bias=False)
        self.W_o     = nn.Linear(d_model, d_model, bias=False)
        self.dropout      = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        self.attn_weights = attn.detach().cpu()
        out  = torch.matmul(self.dropout(attn), V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm    = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, attn, ff, dropout):
        super().__init__()
        self.attn = attn
        self.ff   = ff
        self.res  = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.res[0](x, lambda x: self.attn(x, x, x, mask))
        x = self.res[1](x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, ff, dropout):
        super().__init__()
        self.self_attn  = self_attn
        self.cross_attn = cross_attn
        self.ff         = ff
        self.res = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.res[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.res[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.res[2](x, self.ff)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(d_model)

    def forward(self, x, enc_out, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.src_embed  = src_embed
        self.trg_embed  = trg_embed
        self.src_pos    = src_pos
        self.trg_pos    = trg_pos
        self.projection = projection

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, enc_out, src_mask, trg, trg_mask):
        return self.decoder(self.trg_pos(self.trg_embed(trg)), enc_out, src_mask, trg_mask)

    def project(self, x):
        return self.projection(x)

    def forward(self, src, trg, src_mask, trg_mask):
        return self.project(self.decode(self.encode(src, src_mask), src_mask, trg, trg_mask))


def build_transformer(
    src_vocab_size, trg_vocab_size,
    src_seq_len=100, trg_seq_len=100,
    d_model=256, Nx=4, h=8, dropout=0.1, d_ff=1024
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)
    src_pos   = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos   = PositionalEncoding(d_model, trg_seq_len, dropout)

    enc_layers = nn.ModuleList([
        EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, h, dropout),
                     FeedForwardNetwork(d_model, d_ff, dropout), dropout)
        for _ in range(Nx)
    ])
    dec_layers = nn.ModuleList([
        DecoderBlock(d_model,
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     FeedForwardNetwork(d_model, d_ff, dropout), dropout)
        for _ in range(Nx)
    ])

    model = Transformer(
        Encoder(d_model, enc_layers), Decoder(d_model, dec_layers),
        src_embed, trg_embed, src_pos, trg_pos,
        ProjectionLayer(d_model, trg_vocab_size)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
