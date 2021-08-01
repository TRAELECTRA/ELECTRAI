import numpy as np
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)  # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)     # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)  # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))

class MultiHeadAttentionLayer( nn.Module ) :
    def __init__( self, config ) :
        super().__init__()
        self.valueLinearProjection = nn.Linear( config.hidden_dim, config.hidden_dim )
        self.queryLinearProjection = nn.Linear( config.hidden_dim, config.hidden_dim )
        self.keyLinearProjection = nn.Linear( config.hidden_dim, config.hidden_dim )
        self.dropout = nn.Dropout( config.drop_attn )
        self.n_heads = config.n_heads

    def forward( self, x, mask ) :
        """
        x, query(q), key(k), value(v) : B(batch_size), S(seq_len), D(dim)
        mask : B X S
        """
        # (B, S, H, W)
        q, k, v = self.valueLinearProjection( x ), self.queryLinearProjection( x ), self.keyLinearProjection( x )
        # (B, H, S, W)
        q, k, v = ( x.view( *x.size()[:-1], n_heads, x.shape(-1) / n_head ).transpose( 1, 2 )
                    for x in [q, k, v] )
        #q, k, v = ( split_last( x, ( self.n_heads, -1 ) ).transpose( 1, 2 )
        #            for x in [q, k, v] )
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt( k.size(-1) )
        if mask is not None :
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
            # Make Masked Area in a very small value
            # So we can ignore the masked Value
        scores = self.dropout( torch.F.softmax( scores, dim=-1 ) )
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -transpose-> (B, S, H, W)
        h = ( scores @ v ).transpose( 1, 2 ).contiguous()
        h = h.view( *h.size()[:-2], -1 ) # (B, S, D)

class PositionWiseFeedForward( nn.Module ) :
    def __init__( self, config ) :
        self.relu = nn.ReLU()
        self.forward1 = nn.Linear( config.dim, config.dimff )
        self.forward2 = nn.Linear( config.dimff, config.dim )

    def forward( self, x ) :
        return self.forward2( self.relu( self.forward1( x ) ) )

class Block(nn.Module):
    def __init__(slef, config):
        super().__init__()
        self.attn = MultiHeadAttentionLayer(config)
        self.proj = nn.Linear(config.dim, config.dim)
        self.norm1 = LayerNorm(config)
        self.pwff = PositionWiseFeedForward(config)
        self.norm2 = LayerNorm(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        a = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(a)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h

class Transformer( nn.Module ) :
    def __init__( self, config ) :
        self.embedding = Embeddings( config )
        self.blocks = [ Block( config ) for _ in range( config.num_layers ) ]

    def forward( self, x, seg, mask ) :
        embedding = self.embedding( x, seg )
        for block in self.blocks :
            embedding = block( embedding, mask )