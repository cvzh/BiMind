# Experiment Configuration — BIMIND-LLaMA (LLM Backbone + POS Adapter)

> **Source file:** `BIMIND_LLaMA.py`  
> **Model:** `L3BTwoBrainLLM` with `LLMWithPOSAdapter` on top of a frozen LLaMA-7B backbone

---

## 1. Reproducibility

| Setting | Value |
|---|---|
| Random seed | `0` |
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
| Split `random_state` | `0` |

---

## 3. Feature Engineering

### 3.1 LLM Tokenisation

| Setting | Value |
|---|---|
| Tokeniser | `AutoTokenizer` from `LLM_NAME` |
| Padding token | `eos_token` (added if missing, standard for LLaMA) |
| Max subword length (`max_len`) | `256` |
| Padding strategy | `max_length` with truncation |

### 3.2 POS Features (subword-aligned)

| Setting | Value |
|---|---|
| POS dimension (`POS_DIM`) | `5` |
| POS categories | `VERB/AUX` → 0, `NOUN` → 1, `ADJ` → 2, `ADV` → 3, `OTHER` → 4 |
| Encoding | One-hot per subword token, shape `[N, L, 5]` |
| Alignment | Character-offset mapping from spaCy words to LLM subword tokens |
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

---

## 4. Model Architecture — `L3BTwoBrainLLM`

### 4.1 LLM Backbone (`LLMWithPOSAdapter`)

| Setting | Value |
|---|---|
| Backbone model (`LLM_NAME`) | `meta-llama/Llama-2-7b-hf` |
| Backbone hidden size (`d_model`) | `4096` |
| Backbone frozen | `True` (all LLM parameters non-trainable) |
| LLM dtype | `torch.float16` (fp16 for memory efficiency) |
| Hidden states used | Last layer (`outputs.hidden_states[-1]`) |
| Pooling | Masked mean-pool (padding ignored) |

> Alternative backbones commented in source: `mistralai/Mistral-7B-v0.1`, `microsoft/deberta-v3-base`, `roberta-base`.

### 4.2 AGA Adapter (`LLMWithPOSAdapter`)

**Design:** Query-only V-stream injection. Key bias was removed based on ablation: Q alone = +1.97%, K alone = −1.48%, both combined ≈ 0% (learned cancellation eliminated by query-only path).

| Component | Architecture | Detail |
|---|---|---|
| `pos_query_bias` | `Linear(5, 64) → ReLU → Dropout(0.1) → Linear(64, 4096)` | Maps POS one-hot → residual delta |
| `adapter_gate_q` | `Linear(4096+5, 64) → ReLU → Dropout(0.1) → Linear(64,1) → Sigmoid` | Gated injection per token |
| Gate floor (`min_gate`) | `0.3` | Prevents gate collapse (adapter silenced) |
| Injection formula | `adapted_hidden = hidden + gate_q * q_bias` | Query-only; no cancellation path |

**Ablation mode support:**
- `None` — full AGA (default)
- `"no_gate"` — gate forced to `1.0` (unconditional injection)

### 4.3 No-Experience Head (`z0`)

```
Linear(4096 + 225, 256) → ReLU → Dropout(0.1) → Linear(256, num_classes)
```

### 4.4 Experience Injection (FiLM)

```
know_proj:  Linear(d_know, 4096)
film_gamma: Linear(4096, 4096)
film_beta:  Linear(4096, 4096)
hE = h * (1 + tanh(gamma)) + beta
```

### 4.5 Experience Head (`zE`)

```
Linear(4096 + 225, 256) → ReLU → Dropout(0.1) → Linear(256, num_classes)
```

### 4.6 Fusion

| Method | Architecture |
|---|---|
| `gate` (used in experiments) | `Linear(2·4096 + 2, 64) → ReLU → Linear(64,1) → Sigmoid`; `zf = g·z0 + (1−g)·zE` |
| `logit_avg` | `zf = β·z0 + (1−β)·zE` |
| `poe` | Product-of-Experts on softmax probabilities |
| `agree_head` | `Linear(4·4096, 256) → ReLU → Dropout(0.1) → Linear(256, num_classes)` |

### 4.7 Fixed Model Hyperparameters

| Hyperparameter | Value |
|---|---|
| `dense_units` | `256` |
| `dropout` | `0.1` |
| `adapter_hidden` | `64` |
| `min_gate` | `0.3` |

> No Optuna hyperparameter search is used in the LLM variant. All parameters are fixed.

---

## 5. Training

### 5.1 DataLoader

| Setting | Value |
|---|---|
| Train batch size | `16` |
| Val / Test batch size | `16` |
| Shuffle (train) | `True` |

### 5.2 Optimiser

| Setting | Value |
|---|---|
| Optimiser | `AdamW` (only trainable params, i.e. adapter + heads) |
| Learning rate | `2e-4` |
| Weight decay | `0.01` |
| Gradient clipping | `max_norm = 1.0` |
| Mixed precision | `torch.amp.autocast("cuda")` + `GradScaler` |

### 5.3 Loss Function

| Component | Weight |
|---|---|
| CrossEntropyLoss (fused head `z_fused`) | `1.0` |
| CrossEntropyLoss (no-exp head `z0`) | `0.5` |
| CrossEntropyLoss (exp head `zE`) | `0.5` |
| Symmetric KL agreement regularisation | `λ_agree = 0.05` |
| KB sensitivity loss (`cosine_similarity(hE, h)`) | `λ_kb_sens = 0.1` |
| Gate regularisation `(1 − mean(gate_q))` | `λ_gate_reg = 0.05` |
| Class weights | Balanced: `N / (num_classes × count_per_class)` |

> **KB sensitivity loss** penalises `hE` being too similar to `h` regardless of KB content, which occurs when FiLM biases dominate over retrieved knowledge vectors.  
> **Gate regularisation** prevents gate collapse (`adapter_gate → 0`) that would silence the AGA adapter during training.

### 5.4 Early Stopping & Checkpointing

| Setting | Value |
|---|---|
| Max epochs | `20` |
| Patience | `5` (val loss) |
| Best model checkpoint | `best_llm_model.pth` |
| Save method | Atomic write via temp file + `os.fsync` + size validation (> 1 MB); fallback to `/tmp/` on error |

### 5.5 Knowledge Dropout

| Setting | Value |
|---|---|
| `knowledge_dropout` (train) | `0.3` |
| `knowledge_dropout` (eval) | `0.0` |

### 5.6 Fusion Config

| Setting | Value |
|---|---|
| Fusion mode | `gate` |
| `beta` (logit_avg weight, if used) | `0.5` |

---

## 6. AGA Ablation Study (LLM)

Two variants × two knowledge conditions:

| Variant | `use_adapter` | Gate behaviour |
|---|---|---|
| Full AGA | `True` | Learned gate, floor = 0.3 |
| No AGA | `False` | Adapter bypassed entirely |
| No gate (ablation) | `True`, `ablation_mode="no_gate"` | Gate forced to 1.0 |

| Condition | Knowledge vector |
|---|---|
| A | Active (standard inference) |
| B | Zeroed (primary AGA evaluation — isolates text encoder) |

---

## 7. Leakage Analysis

Three knowledge-base setups to verify the model does not exploit retrieval leakage:

| Setup | Knowledge Base | Purpose |
|---|---|---|
| A — In-domain, deduplicated | Training set minus samples with cosine sim ≥ 0.85 to any test sample | Leak-free in-domain baseline |
| B — External corpus | Wikipedia articles (fetched via REST API) or `MM_COVID.csv` | External-only, no train/test overlap |
| C — Intentional leakage | Training + test set (cheating) | Upper bound; shows what memorisation looks like |

| Leakage setting | Value |
|---|---|
| Deduplication threshold | `0.85` cosine similarity |
| KNN probe top-k | `5` |
| Expected result | Model A ≈ C (robust); KNN C >> A (leakage detectable) |

**KB sensitivity diagnostic:** if avg KB Δ (with-KB acc − zeroed-KB acc) < 2%, FiLM biases dominate over actual retrieved content → retrain with `lambda_kb_sens > 0`.

---

## 8. Evaluation

- Metrics: Accuracy, Weighted F1, per-class Precision/Recall/F1
- Three outputs reported: `No-Experience (z0)`, `Have-Experience (zE)`, `Fused (z_fused)`
- Additional diagnostics: fusion gate statistics (`g ≈ 0` → zE dominant; `g ≈ 1` → z0 dominant), KB Δ

---

## 9. External Tools & Libraries

| Library | Purpose |
|---|---|
| `meta-llama/Llama-2-7b-hf` | Frozen LLM backbone |
| spaCy `en_core_web_md` | Tokenisation, POS tagging, subword alignment |
| `sentence-transformers/all-mpnet-base-v2` | KB embedding & retrieval |
| scikit-learn | TF-IDF, LabelEncoder, metrics |
| PyTorch + Hugging Face `transformers` | Model, training |
| `urllib` + Wikipedia REST API | External KB construction (no extra packages) |
