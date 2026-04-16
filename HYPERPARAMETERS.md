# Experiment Configuration — BIMIND + AGA (Custom Transformer)

> **Source file:** `BIMIND_AGA.py`  
> **Model:** `L3BTwoBrain` with `POSGatedTransformerEncoder` (Adaptive Grammar-Aware Attention)

---

## 1. Reproducibility

| Setting | Value |
|---|---|
| Random seed | `42` |
| `torch.backends.cudnn.deterministic` | `True` |
| `torch.backends.cudnn.benchmark` | `False` |

---

## 2. Dataset

| Setting | Value |
|---|---|
| Dataset file | `ReCOVery.csv` |
| Label column | `label` |
| Text column | `statement` |
| Train / Val / Test split | 80 % / 10 % / 10 % |
| Split `random_state` | `42` |
| Vocabulary `min_freq` | `1` |

---

## 3. Feature Engineering

### 3.1 Text Sequences

| Setting | Value |
|---|---|
| Max sequence length | `200` tokens |
| Tokeniser | spaCy `en_core_web_md` (whitespace/punct removed) |
| Padding token | `<PAD>` (index 0) |
| Unknown token | `<UNK>` (index 1) |

### 3.2 POS Features

| Setting | Value |
|---|---|
| POS dimension (`POS_DIM`) | `5` |
| POS categories | `VERB/AUX` → 0, `NOUN` → 1, `ADJ` → 2, `ADV` → 3, `OTHER` → 4 |
| Encoding | One-hot per token, shape `[N, L, 5]` |
| Tool | spaCy `en_core_web_md` |

### 3.3 Content Features (TF-IDF)

| Setting | Value |
|---|---|
| Statement TF-IDF `max_features` | `150` |
| Verb TF-IDF `max_features` | `75` |
| Combined content dim | `225` |

### 3.4 Knowledge Features (RAG)

| Setting | Value |
|---|---|
| Sentence encoder | `sentence-transformers/all-mpnet-base-v2` |
| Knowledge base | In-domain training split texts |
| Top-k retrieved chunks | `3` |
| Aggregation | Mean of top-k embeddings + max similarity + mean similarity |
| Knowledge feature dim | `d_model + 2` (e.g., 770 for mpnet) |

---

## 4. Model Architecture — `L3BTwoBrain`

### 4.1 Text Encoder

| Component | Detail |
|---|---|
| Embedding | `nn.Embedding(vocab_size, 128)`, scaled by `√128` |
| Positional encoding | Learned absolute PE (`nn.Embedding(5000, 128)`) |
| Transformer encoder | `POSGatedTransformerEncoder` (custom, see §4.2) |
| Pooling | Max-pool over sequence dimension |

### 4.2 `POSGatedTransformerEncoder` / `POSGatedAttentionLayer` (AGA Module)

| Hyperparameter | Value / Search Space |
|---|---|
| Embedding dim (`embed_dim`) | `128` (fixed) |
| Number of heads (`num_heads`) | Optuna: categorical `{2, 4, 8, 16, 32, 64, 128}` |
| Number of layers (`num_layers`) | Fixed at `10` |
| Feed-forward dim (`ff_dim`) | Optuna: `128 – 512`, step `64` |
| Dropout | Optuna: `0.1 – 0.5`, step `0.1` |
| Activation | `GELU` |

**AGA-specific components (per layer):**

| Component | Architecture | Init |
|---|---|---|
| `pos_query_bias` (V-stream scale) | `Linear(5,32) → ReLU → Linear(32, nhead) → LayerNorm(nhead)` | weights × 5 |
| `query_bias_scale` | Scalar parameter | `1.0` |
| `adapter_gate_q` (injection gate) | `Linear(d+5, 64) → ReLU → Dropout → Linear(64,1) → Sigmoid` | — |
| `temperature` (per-head) | `nn.Parameter`, shape `[1, nhead, 1, 1]` | layer 0 → `2.0`, decreasing by `0.5`, floor `0.5` |
| Temperature clamp | `[0.1, 5.0]` | — |
| Gate floor (`min_gate`) | `0.3` | — |

> **Key design choice:** Query-only V-stream injection. Key bias was removed because ablation showed Q alone = +1.97%, K alone = −1.48%, both together ≈ 0% (learned cancellation).

### 4.3 No-Experience Head (`z0`)

```
Linear(128 + 225, dense_units) → ReLU → Dropout → Linear(dense_units, num_classes)
```

| Hyperparameter | Search Space |
|---|---|
| `dense_units` | Optuna: `64 – 256`, step `64` |

### 4.4 Experience Injection (FiLM)

```
know_proj:  Linear(d_know, 128)
film_gamma: Linear(128, 128)
film_beta:  Linear(128, 128)
hE = h * (1 + tanh(gamma)) + beta
```

### 4.5 Experience Head (`zE`)

```
Linear(128 + 225, dense_units) → ReLU → Dropout → Linear(dense_units, num_classes)
```

### 4.6 Fusion

| Method | Architecture |
|---|---|
| `gate` (used in experiments) | `Linear(2·128 + 2, 64) → ReLU → Linear(64,1) → Sigmoid`; `zf = g·z0 + (1−g)·zE` |
| `logit_avg` | `zf = β·z0 + (1−β)·zE` |
| `poe` | Product-of-Experts on softmax probabilities |
| `agree_head` | `Linear(4·128, 256) → ReLU → Dropout(0.1) → Linear(256, num_classes)` |

---

## 5. Training

### 5.1 DataLoader

| Setting | Value |
|---|---|
| Train batch size | `16` |
| Val / Test batch size | `16` |
| Shuffle (train) | `True` |

### 5.2 Hyperparameter Search (Optuna)

| Setting | Value |
|---|---|
| Trials | `15` |
| Direction | Maximize validation accuracy (fused head) |
| Search space | `num_heads`, `num_layers`, `ff_dim`, `dense_units`, `dropout`, `lr` |
| Pruning | Trial pruned if `embed_dim % num_heads ≠ 0` |

### 5.3 Optimiser & Scheduler

| Setting | Value |
|---|---|
| Optimiser | `Adam` |
| Learning rate | Optuna: `1e-5 – 1e-2` (log scale) |
| Gradient clipping | `max_norm = 1.0` |
| Mixed precision | `torch.amp.autocast("cuda")` + `GradScaler` |

### 5.4 Loss Function

| Component | Weight |
|---|---|
| CrossEntropyLoss (fused head `z_fused`) | `1.0` |
| CrossEntropyLoss (no-exp head `z0`) | `0.5` |
| CrossEntropyLoss (exp head `zE`) | `0.5` |
| Symmetric KL agreement regularisation | `λ_agree = 0.05` |
| Attention entropy regularisation (per layer) | Enabled, weight = learnable `entropy_weight` init `0.1` |
| Class weights | Balanced: `N / (num_classes × count_per_class)` |

### 5.5 Early Stopping & Checkpointing

| Setting | Value |
|---|---|
| Max epochs | `50` |
| Patience | `3` (val loss) |
| Best model checkpoint | `best_model.pth` |
| Baseline checkpoint | `best_model_base.pth` |

### 5.6 Knowledge Dropout

| Setting | Value |
|---|---|
| `knowledge_dropout` (train) | `0.3` |
| `knowledge_dropout` (eval) | `0.0` |

### 5.7 Fusion Config

| Setting | Value |
|---|---|
| Fusion mode | `gate` |
| `beta` (logit_avg weight) | `0.5` |

---

## 6. AGA Ablation Study

Four variants × two knowledge conditions:

| Variant | `use_adapter` | `use_query_bias` | `use_temperature` |
|---|---|---|---|
| Full AGA | ✓ | ✓ | ✓ |
| No AGA | ✗ | ✓ | ✓ |
| No query bias | ✓ | ✗ | ✓ |
| No temperature | ✓ | ✓ | ✗ |

| Condition | Knowledge vector |
|---|---|
| A | Active (standard inference) |
| B | Zeroed (primary AGA evaluation) |

---

## 7. Evaluation

- Metrics: Accuracy, Weighted F1, per-class Precision/Recall/F1
- Three outputs reported: `No-Experience (z0)`, `Have-Experience (zE)`, `Fused (z_fused)`
- Additional diagnostics: symmetric-KL agreement, VoX gain, fusion gate statistics

---

## 8. External Tools & Libraries

| Library | Purpose |
|---|---|
| spaCy `en_core_web_md` | Tokenisation, POS tagging |
| `sentence-transformers/all-mpnet-base-v2` | KB embedding & retrieval |
| Optuna | Hyperparameter search |
| scikit-learn | TF-IDF, LabelEncoder, metrics |
| PyTorch | Model, training |
| SHAP | Feature attribution (interpretability) |
