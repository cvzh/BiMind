import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import POS_DIM


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class LearnedAbsolutePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(pos))


# ---------------------------------------------------------------------------
# POS-gated attention
# ---------------------------------------------------------------------------
class POSGatedAttentionLayer(nn.Module):
    """Transformer layer with POS-aware attention bias.

    Key design choices:
    - Learned additive bias on attention *scores* (before softmax) derived from
      query and key POS tags.
    - Per-head learnable temperature to control attention sharpness.
    - Entropy regularisation to prevent attention collapse.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        pos_dim: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pos_query_bias = nn.Sequential(
            nn.Linear(pos_dim, 32), nn.ReLU(), nn.Linear(32, nhead)
        )
        self.pos_key_bias = nn.Sequential(
            nn.Linear(pos_dim, 32), nn.ReLU(), nn.Linear(32, nhead)
        )

        # Higher temperature → more distributed attention
        self.temperature = nn.Parameter(torch.ones(1, nhead, 1, 1))
        self.entropy_weight = nn.Parameter(torch.tensor(0.1))

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # Diagnostics (detached, not part of the graph)
        self.last_attn_weights = None
        self.last_query_gate = None
        self.last_key_gate = None
        self.last_entropy = None

    def forward(
        self,
        src: torch.Tensor,
        pos_feats: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        use_adapter: bool = True,
    ) -> torch.Tensor:
        """
        src:       [B, L, D]
        pos_feats: [B, L, POS_DIM]
        """
        B, L, D = src.shape
        H, head_dim = self.nhead, self.head_dim

        Q = self.q_proj(src).view(B, L, H, head_dim).transpose(1, 2)
        K = self.k_proj(src).view(B, L, H, head_dim).transpose(1, 2)
        V = self.v_proj(src).view(B, L, H, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        if use_adapter:
            q_bias = self.pos_query_bias(pos_feats)          # [B, L, H]
            k_bias = self.pos_key_bias(pos_feats)            # [B, L, H]
            self.last_query_gate = q_bias.detach()
            self.last_key_gate = k_bias.detach()

            q_bias = q_bias.permute(0, 2, 1).unsqueeze(-1)  # [B, H, L, 1]
            k_bias = k_bias.permute(0, 2, 1).unsqueeze(-2)  # [B, H, 1, L]
            attn_scores = attn_scores + q_bias + k_bias

            temp = torch.clamp(self.temperature, min=0.1, max=5.0)
            attn_scores = attn_scores / temp

        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)
        self.last_attn_weights = attn_weights.detach()

        eps = 1e-9
        entropy = (
            -(attn_weights.clamp_min(eps) * attn_weights.clamp_min(eps).log())
            .sum(dim=-1)
        )
        self.last_entropy = entropy.mean().detach()

        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, L, D)
        )
        attn_output = self.out_proj(attn_output)

        src = self.norm1(src + self.dropout1(attn_output))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

    def get_entropy_loss(self) -> torch.Tensor:
        """Entropy regularisation loss (maximise entropy ≡ minimise –H)."""
        if self.last_entropy is not None:
            return -self.entropy_weight * self.last_entropy
        return torch.tensor(0.0)


class POSGatedTransformerEncoder(nn.Module):
    """Stack of POSGatedAttentionLayer with layer-specific initial temperature."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        pos_dim: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                POSGatedAttentionLayer(
                    d_model, nhead, pos_dim, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )
        # Earlier layers get higher temperature to avoid early collapse
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                layer.temperature.fill_(max(2.0 - i * 0.5, 0.5))

    def forward(
        self,
        src: torch.Tensor,
        pos_feats: torch.Tensor,
        src_key_padding_mask=None,
        use_adapter: bool = True,
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(
                src,
                pos_feats,
                src_key_padding_mask=src_key_padding_mask,
                use_adapter=use_adapter,
            )
        return src


# ---------------------------------------------------------------------------
# L³B Two-Brain (custom transformer backbone)
# ---------------------------------------------------------------------------
class L3BTwoBrain(nn.Module):
    """
    Dual-head fake-news classifier with a custom POS-gated transformer encoder.

    - No-exp head : [pooled_text ⊕ content_feats] → z0
    - Exp head    : FiLM-modulated text + knowledge → zE
    - Fusion      : logit_avg | poe | gate | agree_head
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        params: dict,
        additional_feature_dim_noexp: int,
        d_know: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = LearnedAbsolutePE(
            embed_dim, max_len=5000, dropout=params["dropout"]
        )
        self.transformer_encoder = POSGatedTransformerEncoder(
            d_model=embed_dim,
            nhead=params["num_heads"],
            pos_dim=POS_DIM,
            num_layers=params["num_layers"],
            dim_feedforward=params["ff_dim"],
            dropout=params["dropout"],
        )
        self.use_adapter = True

        d_text = embed_dim
        self.d_know = d_know

        self.fc_noexp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes),
        )

        self.know_proj = nn.Linear(d_know, d_text)
        self.film_gamma = nn.Linear(d_text, d_text)
        self.film_beta = nn.Linear(d_text, d_text)

        self.fc_exp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes),
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * d_text + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.agree_head = nn.Sequential(
            nn.Linear(4 * d_text, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        self.register_buffer("T0", torch.tensor(1.0))
        self.register_buffer("TE", torch.tensor(1.0))

    # ------------------------------------------------------------------
    def _encode_text(
        self, x_tokens: torch.Tensor, pos_feats: torch.Tensor, no_checkpoint: bool = False
    ) -> torch.Tensor:
        x = self.embedding(x_tokens) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        pad_mask = x_tokens == 0
        use_adapter = getattr(self, "use_adapter", True)

        if no_checkpoint:
            x = self.transformer_encoder(
                x, pos_feats, src_key_padding_mask=pad_mask, use_adapter=use_adapter
            )
        else:
            x = checkpoint.checkpoint(
                lambda _x, _p: self.transformer_encoder(
                    _x, _p, src_key_padding_mask=pad_mask, use_adapter=use_adapter
                ),
                x,
                pos_feats,
                use_reentrant=False,
            )
        return x.max(dim=1)[0]

    def _inject_experience(self, h: torch.Tensor, know_vec: torch.Tensor) -> torch.Tensor:
        k = self.know_proj(know_vec)
        return h * (1.0 + torch.tanh(self.film_gamma(k))) + self.film_beta(k)

    def forward(
        self,
        x_tokens,
        pos_feats,
        content_feats,
        know_vec,
        fusion: str = "logit_avg",
        beta: float = 0.5,
        knowledge_dropout: float = 0.3,
        no_checkpoint: bool = False,
        train_mode: bool = True,
    ):
        h = self._encode_text(x_tokens, pos_feats, no_checkpoint=no_checkpoint)

        z0 = self.fc_noexp(torch.cat([h, content_feats], dim=1))
        p0 = torch.softmax(z0 / self.T0, dim=-1)

        if train_mode and knowledge_dropout > 0.0:
            mask = (
                torch.rand(know_vec.size(0), device=know_vec.device) > knowledge_dropout
            ).float().unsqueeze(1)
            know_vec = know_vec * mask

        hE = self._inject_experience(h, know_vec)
        zE = self.fc_exp(torch.cat([hE, content_feats], dim=1))
        pE = torch.softmax(zE / self.TE, dim=-1)

        g_val = None
        if fusion == "logit_avg":
            zf = beta * z0 + (1.0 - beta) * zE
        elif fusion == "poe":
            eps = 1e-9
            poe = p0.clamp_min(eps) * pE.clamp_min(eps)
            poe = poe / poe.sum(dim=-1, keepdim=True)
            zf = torch.log(poe.clamp_min(eps))
        elif fusion == "gate":
            ent0 = -(p0.clamp_min(1e-9) * p0.clamp_min(1e-9).log()).sum(-1, keepdim=True)
            entE = -(pE.clamp_min(1e-9) * pE.clamp_min(1e-9).log()).sum(-1, keepdim=True)
            g = self.gate(torch.cat([h, hE, ent0, entE], dim=-1))
            zf = g * z0 + (1.0 - g) * zE
            g_val = g.detach()
        elif fusion == "agree_head":
            feat = torch.cat([h, hE, h * hE, (h - hE).abs()], dim=-1)
            zf = self.agree_head(feat)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        pf = torch.softmax(zf, dim=-1)
        return {
            "z0": z0, "p0": p0,
            "zE": zE, "pE": pE,
            "z_fused": zf, "p_fused": pf,
            "h": h, "hE": hE,
            "gate": g_val,
        }


# ---------------------------------------------------------------------------
# LLM encoder with POS adapter
# ---------------------------------------------------------------------------
class LLMWithPOSAdapter(nn.Module):
    """Frozen LLM backbone with a lightweight POS-aware adapter injected into
    the pooled hidden states."""

    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-2-7b-hf",
        pos_dim: int = 5,
        adapter_hidden: int = 64,
        freeze_llm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.eos_token_id

        self.d_model = self.llm.config.hidden_size

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.pos_query_bias = nn.Sequential(
            nn.Linear(pos_dim, adapter_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, self.d_model),
        )
        self.pos_key_bias = nn.Sequential(
            nn.Linear(pos_dim, adapter_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, self.d_model),
        )
        self.adapter_gate = nn.Sequential(
            nn.Linear(self.d_model + pos_dim, adapter_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, 1),
            nn.Sigmoid(),
        )
        self.adapter_temp = nn.Parameter(torch.tensor(1.0))

        self.last_gate_values = None
        self.use_adapter = True

    def forward(self, input_ids, attention_mask, pos_feats):
        """
        input_ids:     [B, L]
        attention_mask:[B, L]
        pos_feats:     [B, L, POS_DIM]

        Returns:
            pooled: [B, D]
            gate:   [B, L, 1] or None
        """
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]   # [B, L, D]

        if self.use_adapter:
            q_bias = self.pos_query_bias(pos_feats)   # [B, L, D]
            k_bias = self.pos_key_bias(pos_feats)     # [B, L, D]

            gate_input = torch.cat([hidden, pos_feats], dim=-1)
            gate = self.adapter_gate(gate_input) * torch.sigmoid(self.adapter_temp)
            self.last_gate_values = gate.detach()

            adapted_hidden = hidden + gate * (q_bias + k_bias)
        else:
            adapted_hidden = hidden
            gate = None

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (adapted_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled, gate


# ---------------------------------------------------------------------------
# L³B Two-Brain (LLM backbone)
# ---------------------------------------------------------------------------
class L3BTwoBrainLLM(nn.Module):
    """
    Dual-head fake-news classifier with a frozen LLM backbone + POS adapter.

    - No-exp head : [pooled_text ⊕ content_feats] → z0
    - Exp head    : FiLM-modulated text + knowledge → zE
    - Fusion      : gate | logit_avg | poe | agree_head
    """

    def __init__(
        self,
        llm_name: str,
        num_classes: int,
        params: dict,
        additional_feature_dim_noexp: int,
        d_know: int,
        pos_dim: int = 5,
        freeze_llm: bool = True,
    ):
        super().__init__()

        self.encoder = LLMWithPOSAdapter(
            llm_name=llm_name,
            pos_dim=pos_dim,
            adapter_hidden=params.get("adapter_hidden", 64),
            freeze_llm=freeze_llm,
            dropout=params.get("dropout", 0.1),
        )
        d_text = self.encoder.d_model
        self.d_know = d_know

        self.fc_noexp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes),
        )

        self.know_proj = nn.Linear(d_know, d_text)
        self.film_gamma = nn.Linear(d_text, d_text)
        self.film_beta = nn.Linear(d_text, d_text)

        self.fc_exp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes),
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * d_text + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.agree_head = nn.Sequential(
            nn.Linear(4 * d_text, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        self.register_buffer("T0", torch.tensor(1.0))
        self.register_buffer("TE", torch.tensor(1.0))
        self.use_adapter = True

    def _inject_experience(self, h: torch.Tensor, know_vec: torch.Tensor) -> torch.Tensor:
        k = self.know_proj(know_vec)
        return h * (1.0 + torch.tanh(self.film_gamma(k))) + self.film_beta(k)

    def forward(
        self,
        input_ids,
        attention_mask,
        pos_feats,
        content_feats,
        know_vec,
        fusion: str = "gate",
        beta: float = 0.5,
        knowledge_dropout: float = 0.3,
        train_mode: bool = True,
    ):
        self.encoder.use_adapter = self.use_adapter
        h, adapter_gate = self.encoder(input_ids, attention_mask, pos_feats)

        z0 = self.fc_noexp(torch.cat([h, content_feats], dim=1))
        p0 = torch.softmax(z0 / self.T0, dim=-1)

        if train_mode and knowledge_dropout > 0.0:
            mask = (
                torch.rand(know_vec.size(0), device=know_vec.device) > knowledge_dropout
            ).float().unsqueeze(1)
            know_vec = know_vec * mask

        hE = self._inject_experience(h, know_vec)
        zE = self.fc_exp(torch.cat([hE, content_feats], dim=1))
        pE = torch.softmax(zE / self.TE, dim=-1)

        g_val = None
        if fusion == "logit_avg":
            zf = beta * z0 + (1.0 - beta) * zE
        elif fusion == "poe":
            eps = 1e-9
            poe = p0.clamp_min(eps) * pE.clamp_min(eps)
            poe = poe / poe.sum(dim=-1, keepdim=True)
            zf = torch.log(poe.clamp_min(eps))
        elif fusion == "gate":
            ent0 = -(p0.clamp_min(1e-9) * p0.clamp_min(1e-9).log()).sum(-1, keepdim=True)
            entE = -(pE.clamp_min(1e-9) * pE.clamp_min(1e-9).log()).sum(-1, keepdim=True)
            g = self.gate(torch.cat([h, hE, ent0, entE], dim=-1))
            zf = g * z0 + (1.0 - g) * zE
            g_val = g.detach()
        elif fusion == "agree_head":
            feat = torch.cat([h, hE, h * hE, (h - hE).abs()], dim=-1)
            zf = self.agree_head(feat)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        pf = torch.softmax(zf, dim=-1)
        return {
            "z0": z0, "p0": p0,
            "zE": zE, "pE": pE,
            "z_fused": zf, "p_fused": pf,
            "h": h, "hE": hE,
            "gate": g_val,
            "adapter_gate": adapter_gate,
        }
