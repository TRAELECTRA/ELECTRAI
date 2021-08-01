import numpy as np
import torch
import torch.nn as nn

from random import randint, shuffle
from random import random as rand
from transformers import Pipeline

from models.common import Transformer, LayerNorm

class Generator( nn.Module ) :

    def __init__( self, config ) :
        super().__init__()
        self.transformer    = Transformer( config )
        self.fc             = nn.Linear( config.dim, config.dim )
        self.activ1         = nn.Tanh()
        self.linear         = nn.Linear( config.dim, config.dim )
        self.activ2         = nn.ReLU()
        self.norm           = LayerNorm( config )
        self.classifier     = nn.Linear( config.dim, 2 )

        # Decoder
        embed_weight        = self.transformer \
                                  .embedding \
                                  .tok_embed \
                                  .weight
        n_vocab, n_dim      = embed_weight.size()
        self.decoder        = nn.Linear( n_dim, n_vocab, bias=False )
        self.decoder.weight = embed_weight
        self.decoder_bias   = nn.Parameter( torch.zeros( n_vocab) )

    def forward( self, input_idx, segment_idx, input_mask, masked_pos ) :
        h = self.transformer( input_idx, segment_idx, input_mask, masked_pos )
        pooled_h = self.activ1( self.fc( h[:,0] ) )
        masked_pos = masked_pos[:, :, None].expand( -1, -1, h.size( -1 ) )
        h_masked = torch.gather( h, 1, masked_pos )
        h_masked = self.norm( self.active2( self.linear( h_masked ) ) )
        logits_lm = self.decoder( h_masked ) + self.decoder_bias
        logits_clsf = self.classifier( pooled_h )

        return logits_lm, logits_clsf

class PreprocessGenerator(Pipeline):

    def __init__(self, vocab_words, indexer):
        super().__init__()
        self.max_pred = 20  # max tokens of prediction
        self.mask_prob = 0.15  # mask coverage (following mlm)
        self.vocab_words = vocab_words
        self.indexer = indexer  # function from token to token index
        self.max_len = 512

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add special tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_idx = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        input_mask = [1] * len(tokens)

        # For masked language model (MLM)
        masked_tokens, masked_pos = [], []
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))
        candidate_pos = [i for i, token in enumerate(tokens) \
                         if token != '[CLS]' and token != '[SEP]']
        shuffle(candidate_pos)

        for pos in candidate_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_idx = self.indexer(tokens)
        masked_idx = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_idx)
        input_idx.extend([0] * n_pad)
        segment_idx.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        # Zero Padding for Masked Target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_idx.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        return (input_idx, segment_idx, input_mask, masked_idx, masked_pos, masked_weights, is_next)


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words) - 1)
    return vocab_words[i]