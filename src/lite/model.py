"""
CoTT-Lite - modello (configurazione del paper, App. C):
  GRU a 3 layer, hidden 768, LayerNorm tra i layer;
  testa di actionness (lineare, 2 classi, cross-entropy);
  testa di progression (lineare, K bin, histogram loss HL-Gauss).
Il modello e' causale: l'uscita al frame t dipende solo dai frame <= t.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 768
N_LAYERS = 3
N_BINS = 20               # bin della progression in [0, 1]
SIGMA_FACTOR = 0.75       # sigma della HL-Gauss = 0.75 * larghezza di un bin


class CoTTLite(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=N_LAYERS, n_bins=N_BINS, dropout=0.1):
        super().__init__()
        self.grus = nn.ModuleList([nn.GRU(in_dim if i == 0 else hidden, hidden, batch_first=True)
                                   for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.actionness_head = nn.Linear(hidden, 2)
        self.progression_head = nn.Linear(hidden, n_bins)
        self.register_buffer("bin_centers", (torch.arange(n_bins) + 0.5) / n_bins)

    def forward(self, x):                        # x: [B, L, D]
        h = x
        for gru, norm in zip(self.grus, self.norms):
            h, _ = gru(h)
            h = self.dropout(norm(h))
        return self.actionness_head(h), self.progression_head(h)   # [B, L, 2], [B, L, K]

    def decode(self, act_logits, prog_logits):
        """Probabilita' di azione e progression attesa (media sui bin)."""
        p_action = act_logits.softmax(-1)[..., 1]
        progression = (prog_logits.softmax(-1) * self.bin_centers).sum(-1)
        return p_action, progression


def hl_gauss_targets(progression, n_bins=N_BINS, sigma_factor=SIGMA_FACTOR):
    """Distribuzione target sui bin: gaussiana centrata sul valore vero (Imani et al., 2024)."""
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=progression.device)
    sigma = sigma_factor / n_bins
    cdf = 0.5 * (1 + torch.erf((edges - progression.unsqueeze(-1)) / (sigma * math.sqrt(2))))
    probs = cdf[..., 1:] - cdf[..., :-1]
    return probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)


def cott_lite_loss(act_logits, prog_logits, actionness, progression, action_mask, valid_mask):
    """Cross-entropy sull'actionness (frame validi) + histogram loss sulla progression (frame di azione)."""
    ce = F.cross_entropy(act_logits[valid_mask], actionness[valid_mask])
    prog_mask = valid_mask & action_mask
    if prog_mask.any():
        target = hl_gauss_targets(progression[prog_mask], prog_logits.shape[-1])
        hl = -(target * F.log_softmax(prog_logits[prog_mask], dim=-1)).sum(-1).mean()
    else:
        hl = prog_logits.sum() * 0.0
    return ce + hl, ce.detach(), hl.detach()
