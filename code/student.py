"""Modern pre-norm transformer for the MP1 board (self-contained; only this file is hashed).

Everything is configuration driven and every added key carries a modern default, so a config that
holds only the baseline keys (vocab/width/heads/depth/context) still builds the modern stack:

    norm    : 'rmsnorm' | 'layernorm'          (default rmsnorm)
    mlp     : 'swiglu'  | 'gelu'               (default swiglu, hidden = 8/3 * width)
    qk_norm : bool                             (default True, RMSNorm over each head dimension)
    tie     : bool                             (default True, embedding and head share one matrix)
    bias    : bool                             (default False, no bias in the hidden linears)
    dropout : float                            (default 0.0; applied after the embeddings and on both
                                                residual branches. nn.Dropout is the identity in eval
                                                mode, so scoring is unaffected whenever it is on.)
    mtp_k   : int                              (default 0, no auxiliary heads) number of multi-token
                                                prediction heads used during training only. Head j
                                                predicts the token j+2 positions ahead from the hidden
                                                state at each position; they are built only when
                                                mtp_k > 0, are zero-initialised, and never take part
                                                in forward() or predict_log_probs().

Relative to the classroom baseline this swaps LayerNorm for RMSNorm, GELU-4x for SwiGLU at an equal
3 * width * hidden parameter budget, normalises q and k before attention, and zero-initialises the
attention output projection and the MLP down projection so every block starts as the identity map.
Positions stay learned-absolute: the board fixes context=256, where RoPE buys little.
"""
import torch
from torch import nn
from torch.nn import functional as F

DEFAULTS = {'norm': 'rmsnorm', 'mlp': 'swiglu', 'qk_norm': True, 'tie': True, 'bias': False,
            'dropout': 0.0, 'mtp_k': 0, 'init_std': .02, 'hidden_multiple': 8}


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        dtype = x.dtype
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def make_norm(norm, width):
    if norm == 'rmsnorm':
        return RMSNorm(width)
    if norm == 'layernorm':
        return nn.LayerNorm(width)
    raise ValueError(f'Unknown norm: {norm}')


def swiglu_hidden(width, multiple):
    """8/3 * width rounded to `multiple`, so three SwiGLU matrices cost the same as GELU at 4x."""
    return max(multiple, int(round(8 * width / 3 / multiple)) * multiple)


def make_dropout(probability):
    return nn.Dropout(float(probability)) if float(probability) > 0 else nn.Identity()


class Attention(nn.Module):
    def __init__(self, width, heads, qk_norm, bias):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads.')
        self.heads = heads
        self.width = width
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        self.q_norm = RMSNorm(width // heads) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(width // heads) if qk_norm else nn.Identity()

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden, bias):
        super().__init__()
        self.gate, self.up, self.down = (nn.Linear(width, hidden, bias=bias),
                                         nn.Linear(width, hidden, bias=bias),
                                         nn.Linear(hidden, width, bias=bias))

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GELUMLP(nn.Module):
    def __init__(self, width, bias):
        super().__init__()
        self.up, self.down = nn.Linear(width, 4 * width, bias=bias), nn.Linear(4 * width, width, bias=bias)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, width, heads, config):
        super().__init__()
        self.norm1 = make_norm(config['norm'], width)
        self.norm2 = make_norm(config['norm'], width)
        self.attn = Attention(width, heads, config['qk_norm'], config['bias'])
        self.mlp = (SwiGLU(width, swiglu_hidden(width, config['hidden_multiple']), config['bias'])
                    if config['mlp'] == 'swiglu' else GELUMLP(width, config['bias']))
        self.attn_dropout = make_dropout(config['dropout'])
        self.mlp_dropout = make_dropout(config['dropout'])

    def forward(self, x):
        x = x + self.attn_dropout(self.attn(self.norm1(x)))
        return x + self.mlp_dropout(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = DEFAULTS | dict(config)
        self.config = dict(config)
        self.context = int(config['context'])
        extra = set(self.config) - set(DEFAULTS)
        if extra - {'vocab', 'width', 'heads', 'depth', 'context'}:
            raise ValueError(f'Unknown config keys: {sorted(extra)}')
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([Block(width, config['heads'], config) for _ in range(config['depth'])])
        self.norm = make_norm(config['norm'], width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.embed_dropout = make_dropout(config['dropout'])
        if int(config['mtp_k']) > 0:
            # nn.Linear's constructor draws from the global RNG, so the heads are built with the
            # stream saved and restored: an MTP run must start from the same initial weights as its
            # control, otherwise the screening comparison carries initialisation noise.
            rng_state = torch.random.get_rng_state()
            self.mtp_heads = nn.ModuleList([nn.Linear(width, config['vocab'], bias=False)
                                            for _ in range(int(config['mtp_k']))])
            torch.random.set_rng_state(rng_state)
        self.apply(self.initialize)
        if config['tie']:
            self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def zero_outputs(self):
        """Start every block as the identity map, as in the residual-zero-initialised literature."""
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.mlp.down.weight)
        for head in getattr(self, 'mtp_heads', ()):
            nn.init.zeros_(head.weight)

    def forward_with_mtp(self, ids):
        """Training-only interface: the main logits plus one logits tensor per auxiliary head.

        mtp_logits[j][b, t] is head j's prediction of ids[b, t+j+2], i.e. of the token j+2 positions
        ahead of position t, so head 0 extends the ordinary one-token-ahead objective by one more
        step. Positions with no such target inside the window are simply absent from the tensor's
        usable span and are dropped by the training loss, which counts each head over its own span.
        """
        hidden = self.features(ids)
        return self.head(hidden), [head(hidden) for head in getattr(self, 'mtp_heads', ())]

    def features(self, ids):
        x = self.embed_dropout(self.token(ids)) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        Strictly causal and window independent: a prediction at position t reads ids[:, :t+1] only,
        no module keeps state between calls and nothing couples the rows of a batch.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    model = GPT(config)
    model.zero_outputs()
    return model
