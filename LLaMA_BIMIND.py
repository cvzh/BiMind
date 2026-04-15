
import os
import gc
import math
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

nlp = spacy.load("en_core_web_md")

sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to("cuda" if torch.cuda.is_available() else "cpu")

# Tokenization & utilities
def tokenize(text):
    doc = nlp.make_doc(text)
    return [t.text.lower() for t in doc if not (t.is_space or t.is_punct)]

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def text_to_sequence(text, vocab, max_len):
    seq = [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)]
    return seq[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(seq))

def extract_verbs(text):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.pos_ == "VERB"]

def link_to_external_knowledge(texts, kb_embeddings: torch.Tensor, top_k: int = 5) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    device = kb_embeddings.device
    q = sentence_model.encode(texts, convert_to_tensor=True, device=device)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = torch.nn.functional.normalize(q, p=2, dim=1)  # for cosine similarity

    scores = q @ kb_embeddings.T
    k = min(top_k, kb_embeddings.size(0))
    vals, idx = torch.topk(scores, k=k, dim=1)

    sel = kb_embeddings.index_select(0, idx.reshape(-1)).reshape(q.size(0), k, -1)
    agg = sel.mean(dim=1)  # [B, D]

    max_sim = vals.max(dim=1).values.unsqueeze(1)
    mean_sim = vals.mean(dim=1, keepdim=True)

    out = torch.cat([agg, max_sim, mean_sim], dim=1)
    return out.detach().cpu().numpy().astype("float32")

POS2ID = {"VERB": 0, "AUX": 0, "NOUN": 1, "ADJ": 2, "ADV": 3}
POS_DIM = 5

def pos_onehot_from_tag(tag: str):
    v = [0] * POS_DIM
    idx = POS2ID.get(tag, 4)
    v[idx] = 1
    return v

# Precompute POS matrices
def pos_mats_for_texts(texts, max_len):
    mats = []
    for doc in nlp.pipe(texts, batch_size=256, n_process=1):
        # toks = [tok.pos_ for tok in doc][:max_len]
        toks = [tok.pos_ for tok in doc if not (tok.is_space or tok.is_punct)]
        toks = toks[:max_len]
        rows = [pos_onehot_from_tag(tag) for tag in toks]
        while len(rows) < max_len:
            rows.append(pos_onehot_from_tag("OTHER"))
        mats.append(np.asarray(rows, dtype=np.float32))
    return np.stack(mats, axis=0)

def summarize_gates(gates_np, name="Gate"):
    if gates_np is None or gates_np.size == 0:
        print(f"{name}: no gate values collected.")
        return
    g = gates_np.ravel()
    mean = float(np.mean(g))
    median = float(np.median(g))
    pct_low = 100.0 * float(np.mean(g < 0.3))
    pct_high = 100.0 * float(np.mean(g > 0.7))
    print(f"{name} — mean: {mean:.4f} | median: {median:.4f} | "
          f"%<0.3: {pct_low:.2f}% | %>0.7: {pct_high:.2f}%")

# Dataset
class NewsDataset(Dataset):
    def __init__(self, sequences, pos_feats, content_features, knowledge_features, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.pos_feats = torch.tensor(pos_feats, dtype=torch.float)           # [N, L, POS_DIM]
        self.content_features = torch.tensor(content_features, dtype=torch.float)
        self.knowledge_features = torch.tensor(knowledge_features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.pos_feats[idx],
            self.content_features[idx],
            self.knowledge_features[idx],
            self.labels[idx]
        )

# Encoder
class LearnedAbsolutePE(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]
        x = x + self.pe(pos)
        return self.dropout(x)

class POSGatedAttentionLayer(nn.Module):
    """
    Attention layer with POS-aware attention bias that:
    1. Adds learned bias to attention scores BEFORE softmax
    2. Prevents attention collapse by encouraging distribution
    3. Uses temperature scaling to control sharpness
    """
    def __init__(self, d_model, nhead, pos_dim, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Manual Q, K, V projections (needed to inject bias before softmax)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # POS-aware attention bias: adds to attention scores before softmax
        # Query POS → how this token distributes attention
        self.pos_query_bias = nn.Sequential(
            nn.Linear(pos_dim, 32),
            nn.ReLU(),
            nn.Linear(32, nhead),  # Per-head bias
        )
        
        # Key POS → how much this token attracts attention
        self.pos_key_bias = nn.Sequential(
            nn.Linear(pos_dim, 32),
            nn.ReLU(),
            nn.Linear(32, nhead),  # Per-head bias
        )
        
        # Learnable temperature per head (higher = more distributed)
        self.temperature = nn.Parameter(torch.ones(1, nhead, 1, 1) * 1.0)
        
        # Anti-collapse: entropy regularization strength
        self.entropy_weight = nn.Parameter(torch.tensor(0.1))
        
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
        # Tracking
        self.last_attn_weights = None
        self.last_query_gate = None
        self.last_key_gate = None
        self.last_gate_values = None
        self.last_entropy = None

    def forward(self, src, pos_feats, src_mask=None, src_key_padding_mask=None, use_adapter=True):
        """
        src: [B, L, D]
        pos_feats: [B, L, POS_DIM]
        """
        B, L, D = src.shape
        H = self.nhead
        head_dim = self.head_dim
        
        # Project Q, K, V
        Q = self.q_proj(src).view(B, L, H, head_dim).transpose(1, 2)  # [B, H, L, head_dim]
        K = self.k_proj(src).view(B, L, H, head_dim).transpose(1, 2)  # [B, H, L, head_dim]
        V = self.v_proj(src).view(B, L, H, head_dim).transpose(1, 2)  # [B, H, L, head_dim]
        
        # Compute raw attention scores
        scale = math.sqrt(head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, L, L]
        
        if use_adapter:
            # Compute POS-based attention biases
            q_bias = self.pos_query_bias(pos_feats)  # [B, L, H]
            k_bias = self.pos_key_bias(pos_feats)    # [B, L, H]
            
            self.last_query_gate = q_bias.detach()
            self.last_key_gate = k_bias.detach()
            
            # Reshape for broadcasting: q_bias affects rows, k_bias affects columns
            q_bias = q_bias.permute(0, 2, 1).unsqueeze(-1)   # [B, H, L, 1]
            k_bias = k_bias.permute(0, 2, 1).unsqueeze(-2)   # [B, H, 1, L]
            
            # Add biases to attention scores (before softmax!)
            # This encourages attention to/from content words
            attn_scores = attn_scores + q_bias + k_bias
            
            # Apply temperature scaling (higher temp = more distributed)
            temp = torch.clamp(self.temperature, min=0.1, max=5.0)
            attn_scores = attn_scores / temp
        
        # Apply padding mask
        if src_key_padding_mask is not None:
            # src_key_padding_mask: [B, L], True = pad
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        attn_weights = self.dropout_attn(attn_weights)
        
        self.last_attn_weights = attn_weights.detach()
        
        # Compute entropy for monitoring
        eps = 1e-9
        entropy = -(attn_weights.clamp_min(eps) * attn_weights.clamp_min(eps).log()).sum(dim=-1)
        self.last_entropy = entropy.mean().detach()
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        attn_output = self.out_proj(attn_output)
        
        # Residual + norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def get_entropy_loss(self):
        """Returns entropy regularization loss to encourage distributed attention."""
        if self.last_entropy is not None:
            # Maximize entropy = minimize negative entropy
            return -self.entropy_weight * self.last_entropy
        return 0.0

class POSGatedTransformerEncoder(nn.Module):
    """Stacks multiple POSGatedAttentionLayer with layer-specific temperature."""
    def __init__(self, d_model, nhead, pos_dim, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            POSGatedAttentionLayer(d_model, nhead, pos_dim, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Force higher temperature in early layers to prevent collapse
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'temperature'):
                    # Layer 0 gets high temp (2.0), decreases for later layers
                    init_temp = 2.0 - (i * 0.5)
                    layer.temperature.fill_(max(init_temp, 0.5))
    
    def forward(self, src, pos_feats, src_key_padding_mask=None, use_adapter=True):
        for layer in self.layers:
            src = layer(src, pos_feats, src_key_padding_mask=src_key_padding_mask, use_adapter=use_adapter)
        return src

# Dual heads
class L3BTwoBrain(nn.Module):
    """
    - No-exp head: [pooled_text ⊕ content_feats] → z0
    - Exp head: FiLM-injected text with knowledge → [pooled_text_E ⊕ content_feats] → zE
    Fusion: logit_avg | poe | gate | agree_head
    """
    def __init__(self, vocab_size, embed_dim, num_classes, params,
                 additional_feature_dim_noexp, d_know):
        super().__init__()

        # Text encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = LearnedAbsolutePE(embed_dim, max_len=5000, dropout=params["dropout"])
        self.transformer_encoder = POSGatedTransformerEncoder(
            d_model=embed_dim,
            nhead=params["num_heads"],
            pos_dim=POS_DIM,
            num_layers=params["num_layers"],
            dim_feedforward=params["ff_dim"],
            dropout=params["dropout"]
        )

        self.use_adapter = True  # ADD THIS LINE

        d_text = embed_dim
        self.d_know = d_know

        # No-exp head (content)
        self.fc_noexp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes)
        )

        # Experience injection (FiLM)
        self.know_proj  = nn.Linear(d_know, d_text)
        self.film_gamma = nn.Linear(d_text, d_text)
        self.film_beta  = nn.Linear(d_text, d_text)

        # Exp head
        self.fc_exp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes)
        )

        # Fusion
        self.gate = nn.Sequential(
            nn.Linear(2 * d_text + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.agree_head = nn.Sequential(
            nn.Linear(4 * d_text, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        self.register_buffer("T0", torch.tensor(1.0))
        self.register_buffer("TE", torch.tensor(1.0))

    def _encode_text(self, x_tokens, pos_feats, no_checkpoint=False):
        x = self.embedding(x_tokens) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)

        pad_id = 0
        pad_mask = x_tokens == pad_id

        use_adapter = getattr(self, "use_adapter", True)

        if no_checkpoint:
            x = self.transformer_encoder(x, pos_feats, src_key_padding_mask=pad_mask, use_adapter=use_adapter)
        else:
            x = checkpoint.checkpoint(
                lambda _x, _p: self.transformer_encoder(_x, _p, src_key_padding_mask=pad_mask, use_adapter=use_adapter),
                x, pos_feats,
                use_reentrant=False,
            )

        pooled = x.max(dim=1)[0]
        return pooled

    def _inject_experience(self, h, know_vec):
        k = self.know_proj(know_vec)
        gamma = self.film_gamma(k)
        beta  = self.film_beta(k)
        return h * (1.0 + torch.tanh(gamma)) + beta

    def forward(self,
                x_tokens,
                pos_feats,
                content_feats,
                know_vec,
                fusion="logit_avg",
                beta=0.5,
                knowledge_dropout=0.3,
                no_checkpoint=False,
                train_mode=True):
        # self.attention_scores = []
        h = self._encode_text(x_tokens, pos_feats, no_checkpoint=no_checkpoint)  # [B, d]

        # Head 0
        z0 = self.fc_noexp(torch.cat([h, content_feats], dim=1))
        p0 = torch.softmax(z0 / self.T0, dim=-1)

        # Knowledge dropout
        if train_mode and knowledge_dropout > 0.0:
            mask = (torch.rand(know_vec.size(0), device=know_vec.device) > knowledge_dropout).float().unsqueeze(1)
            know_vec = know_vec * mask

        # Head E
        hE = self._inject_experience(h, know_vec)
        zE = self.fc_exp(torch.cat([hE, content_feats], dim=1))
        pE = torch.softmax(zE / self.TE, dim=-1)
        g_val = None # 1)
        # Fusion
        if fusion == "logit_avg":
            zf = beta * z0 + (1.0 - beta) * zE
        elif fusion == "poe":
            eps = 1e-9
            poe = (p0.clamp_min(eps) * pE.clamp_min(eps))
            poe = poe / poe.sum(dim=-1, keepdim=True)
            zf = torch.log(poe.clamp_min(eps))
        elif fusion == "gate":
            ent0 = (-(p0.clamp_min(1e-9) * p0.clamp_min(1e-9).log()).sum(-1, keepdim=True))
            entE = (-(pE.clamp_min(1e-9) * pE.clamp_min(1e-9).log()).sum(-1, keepdim=True))
            g = self.gate(torch.cat([h, hE, ent0, entE], dim=-1))
            zf = g * z0 + (1.0 - g) * zE
            g_val = g.detach() # 1)
        elif fusion == "agree_head":
            feat = torch.cat([h, hE, h * hE, (h - hE).abs()], dim=-1)
            zf = self.agree_head(feat)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        pf = torch.softmax(zf, dim=-1)
        return {"z0": z0, "p0": p0, "zE": zE, "pE": pE, "z_fused": zf, "p_fused": pf, "h": h, "hE": hE, "gate": g_val} # 1) g_val

# Feature preparation
def prepare_features(
    data_df, vocab, max_seq_length, kb_embeddings,
    tfidf_vectorizer=None, verb_vectorizer=None, fit_vectorizers=False):

    # sequences
    sequences = [text_to_sequence(text, vocab, max_seq_length) for text in data_df["statement"]]

    # POS matrices
    pos_mats = pos_mats_for_texts(data_df["statement"].tolist(), max_seq_length)
    # pos_mats = np.stack(mats, axis=0)  # [N, L, POS_DIM]

    if fit_vectorizers or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=150)
        tfidf_statement = tfidf_vectorizer.fit_transform(data_df["statement"]).toarray()
    else:
        tfidf_statement = tfidf_vectorizer.transform(data_df["statement"]).toarray()

    verbs_corpus = []
    for text in data_df["statement"]:
        ex = extract_verbs(text)
        verbs_corpus.append(" ".join(ex) if ex else "no_verb")
    if fit_vectorizers or verb_vectorizer is None:
        verb_vectorizer = TfidfVectorizer(max_features=75)
        tfidf_verbs = verb_vectorizer.fit_transform(verbs_corpus).toarray()
    else:
        tfidf_verbs = verb_vectorizer.transform(verbs_corpus).toarray()

    content_features = np.hstack([tfidf_statement, tfidf_verbs])

    # Knowledge features
    knowledge_features = link_to_external_knowledge(
        data_df["statement"].tolist(), kb_embeddings=kb_embeddings, top_k=3
    )

    print(f"POS mats: {pos_mats.shape} | TF-IDF Statement: {tfidf_statement.shape} | TF-IDF Verbs: {tfidf_verbs.shape} | "
          f"Content: {content_features.shape} | Knowledge: {knowledge_features.shape}")

    return sequences, pos_mats, content_features, knowledge_features, tfidf_vectorizer, verb_vectorizer

# Train
def train_model(train_loader, val_loader, model, criterion, optimizer, config):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler("cuda")
    fusion = config.get("fusion", "logit_avg")
    beta = config.get("beta", 0.5)
    know_drop = config.get("knowledge_dropout", 0.3)
    lambda_agree = config.get("lambda_agree", 0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0.0
        running_correct = 0
        running_total = 0

        for seqs, posf, cont, know, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            seqs = seqs.to(config["device"])
            posf = posf.to(config["device"])
            cont = cont.to(config["device"])
            know = know.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(seqs, posf, cont, know, fusion=fusion, beta=beta,
                            knowledge_dropout=know_drop, train_mode=True)
                loss = criterion(out["z_fused"], labels) \
                       + 0.5 * criterion(out["z0"], labels) \
                       + 0.5 * criterion(out["zE"], labels)
                p0 = out["p0"].clamp_min(1e-9); pE = out["pE"].clamp_min(1e-9)
                symkl = 0.5 * ((p0 * (p0 / pE).log()).sum(-1) +
                               (pE * (pE / p0).log()).sum(-1)).mean()
                loss = loss + lambda_agree * symkl

                # ADD: Entropy regularization to prevent attention collapse
                for layer in model.transformer_encoder.layers:
                    if hasattr(layer, 'get_entropy_loss'):
                        loss = loss + layer.get_entropy_loss()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            running_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
            running_total += labels.size(0)

            del seqs, posf, cont, know, labels, out, loss
            torch.cuda.empty_cache()

        train_epoch_loss = epoch_loss / max(len(train_loader), 1)
        train_epoch_acc = (running_correct / max(running_total, 1)) if running_total else 0.0
        print(f"Epoch {epoch+1} Training Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc*100:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_gates = []
        with torch.no_grad():
            for seqs, posf, cont, know, labels in val_loader:
                seqs = seqs.to(config["device"])
                posf = posf.to(config["device"])
                cont = cont.to(config["device"])
                know = know.to(config["device"])
                labels = labels.to(config["device"])
                out = model(seqs, posf, cont, know, fusion=fusion, beta=beta,
                            knowledge_dropout=0.0, train_mode=False)
                val_loss += float(criterion(out["z_fused"], labels).detach().cpu())

                val_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
                val_total += labels.size(0)

                if out.get("gate", None) is not None:
                    val_gates.append(out["gate"].cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        val_acc = (val_correct / max(val_total, 1)) if val_total else 0.0
        print(f"Validation Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_acc)

        if fusion == "gate" and len(val_gates) > 0:
            gates_np = np.concatenate(val_gates, axis=0)
            summarize_gates(gates_np, name="Val Gate")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config["best_model_path"])
            print(f"✅ Saved best to {config['best_model_path']}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print("Early stopping.")
                break

    return history

# Test
def test_model(test_loader, model, config, feature_names, vocab):
    device = config["device"]
    fusion = config.get("fusion", "logit_avg")
    beta = config.get("beta", 0.5)

    model.to(device)
    model.eval()

    all_labels, y0, yE, yF = [], [], [], []
    agree_vals, vox_gains = [], []
    gate_vals = []

    with torch.no_grad():
        for seqs, posf, cont, know, labels in test_loader:
            seqs = seqs.to(device); posf = posf.to(device)
            cont = cont.to(device); know = know.to(device); labels = labels.to(device)
            out = model(seqs, posf, cont, know, fusion=fusion, beta=beta, knowledge_dropout=0.0, train_mode=False)

            all_labels.extend(labels.cpu().numpy())
            y0.extend(out["p0"].argmax(-1).cpu().numpy())
            yE.extend(out["pE"].argmax(-1).cpu().numpy())
            yF.extend(out["p_fused"].argmax(-1).cpu().numpy())

            # Agreement (sym-KL)
            p0 = out["p0"].clamp_min(1e-9); pE = out["pE"].clamp_min(1e-9)
            skl = 0.5 * ((p0 * (p0 / pE).log()).sum(-1) + (pE * (pE / p0).log()).sum(-1))
            agree_vals.extend(skl.cpu().numpy())

            # VoX: correct-class logit gain
            gy = out["zE"].gather(1, labels.view(-1,1)) - out["z0"].gather(1, labels.view(-1,1))
            vox_gains.extend(gy.squeeze(1).cpu().numpy())

            if out.get("gate", None) is not None:
              gate_vals.append(out["gate"].cpu().numpy())

    y_true = np.array(all_labels); y0 = np.array(y0); yE = np.array(yE); yF = np.array(yF)

    def report(name, yhat):
        acc = accuracy_score(y_true, yhat)
        f1w = f1_score(y_true, yhat, average="weighted")
        print(f"✅ {name} — Acc: {acc*100:.2f}% | F1(w): {f1w*100:.2f}%")
        print(classification_report(y_true, yhat, digits=4))

    print("\n=== L³B Two-Brain Evaluation ===")
    report("No-Experience", y0)
    report("Have-Experience", yE)
    report("Fused", yF)

    print(f"Agreement (sym-KL) mean: {np.mean(agree_vals):.4f} | median: {np.median(agree_vals):.4f}")
    print(f"VoX gain mean: {np.mean(vox_gains):.4f} | median: {np.median(vox_gains):.4f} | positive%: {100*np.mean(np.array(vox_gains)>0):.2f}%")

    if fusion == "gate" and len(gate_vals) > 0:
        gates_np = np.concatenate(gate_vals, axis=0)
        summarize_gates(gates_np, name="Test Gate")

    os.makedirs("reports", exist_ok=True)
    with open("reports/test_results.txt", "w") as f:
        f.write("Two-Brain Results\n")
        f.write(f"Agreement sym-KL mean: {np.mean(agree_vals):.4f}\n")
        f.write(f"VoX gain mean: {np.mean(vox_gains):.4f}\n")

# Token alignment utility
def align_pos_to_subwords(texts, tokenizer, nlp, max_len=512):
    """
    Align spaCy POS tags to LLM subword tokens.
    Returns: input_ids, attention_mask, aligned_pos_feats
    """
    all_input_ids = []
    all_attention_masks = []
    all_pos_feats = []
    
    for text in texts:
        # Get spaCy POS tags per word
        doc = nlp(text)
        word_pos = [(tok.text, tok.pos_, tok.idx, tok.idx + len(tok.text)) 
                    for tok in doc if not (tok.is_space or tok.is_punct)]
        
        # Tokenize with LLM tokenizer
        encoding = tokenizer(
            text, 
            return_offsets_mapping=True, 
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)
        
        # Build character-to-POS mapping
        char_to_pos = {}
        for word, pos, start, end in word_pos:
            for c in range(start, end):
                char_to_pos[c] = pos
        
        # Align each subword token to its POS
        pos_ids = []
        for start, end in offsets:
            start, end = start.item(), end.item()
            if start == end:  # Special token ([CLS], [SEP], [PAD])
                pos_ids.append("OTHER")
            else:
                # Use the POS of the first character in this subword
                pos = char_to_pos.get(start, "OTHER")
                pos_ids.append(pos)
        
        # Convert to one-hot
        pos_feats = torch.tensor([pos_onehot_from_tag(p) for p in pos_ids], dtype=torch.float)
        
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_pos_feats.append(pos_feats)
    
    return (
        torch.stack(all_input_ids),
        torch.stack(all_attention_masks),
        torch.stack(all_pos_feats)
    )

class LLMWithPOSAdapter(nn.Module):
    """
    Uses frozen LLM as backbone, injects POS-aware signals via adapter.
    """
    
    def __init__(self, llm_name="meta-llama/Llama-2-7b-hf", pos_dim=5, # meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1
                 adapter_hidden=64, freeze_llm=True, dropout=0.1):
        super().__init__()
        
        # Load LLaMA as causal LM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            # use_auth_token=True,  # Required for LLaMA
            torch_dtype=torch.float16,  # Use fp16 to save memory
            # device_map="auto"  # Automatic device placement
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            # use_auth_token=True  # Required for LLaMA
        )
        # Add padding token (LLaMA doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.d_model = self.llm.config.hidden_size
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # POS-aware adapter
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
        
        # Adapter injection gate (learned mixing)
        self.adapter_gate = nn.Sequential(
            nn.Linear(self.d_model + pos_dim, adapter_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, 1),
            nn.Sigmoid()
        )
        
        # Learnable temperature for controlling adapter strength
        self.adapter_temp = nn.Parameter(torch.tensor(1.0))
        
        # For tracking
        self.last_gate_values = None
        self.use_adapter = True

    def forward(self, input_ids, attention_mask, pos_feats):
        """
        input_ids: [B, L] from LLM tokenizer
        attention_mask: [B, L] 
        pos_feats: [B, L, POS_DIM] aligned to LLM tokens
        """
        # # Get LLM hidden states
        # outputs = self.llm(
        #     input_ids=input_ids, 
        #     attention_mask=attention_mask,
        #     output_hidden_states=True
        # )
        # hidden = outputs.last_hidden_state  # [B, L, D]
        # Get hidden states from LLaMA
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # LLaMA uses .hidden_states instead of .last_hidden_state
        hidden = outputs.hidden_states[-1]  # Last layer
        
        if self.use_adapter:
            # Compute POS-aware modulation
            q_bias = self.pos_query_bias(pos_feats)  # [B, L, D]
            k_bias = self.pos_key_bias(pos_feats)    # [B, L, D]
            
            # Gated injection
            gate_input = torch.cat([hidden, pos_feats], dim=-1)
            gate = self.adapter_gate(gate_input)  # [B, L, 1]
            
            # Temperature-scaled gate
            gate = gate * torch.sigmoid(self.adapter_temp)
            
            self.last_gate_values = gate.detach()
            
            # Inject adapter signals into LLM embeddings
            adapted_hidden = hidden + gate * (q_bias + k_bias)
        else:
            adapted_hidden = hidden
            gate = None
        
        # Masked pooling (ignore padding)
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        masked_hidden = adapted_hidden * mask
        pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, D]
        
        return pooled, gate

class L3BTwoBrainLLM(nn.Module):
    """
    Two-Brain architecture with LLM backbone + POS adapter.
    - No-exp head: [pooled_text ⊕ content_feats] → z0
    - Exp head: FiLM-injected text with knowledge → zE
    - Fusion: gate | logit_avg | poe
    """
    def __init__(self, llm_name, num_classes, params,
                 additional_feature_dim_noexp, d_know, pos_dim=5, freeze_llm=True):
        super().__init__()
        
        # LLM encoder with POS adapter
        self.encoder = LLMWithPOSAdapter(
            llm_name=llm_name,
            pos_dim=pos_dim,
            adapter_hidden=params.get("adapter_hidden", 64),
            freeze_llm=freeze_llm,
            dropout=params.get("dropout", 0.1)
        )
        d_text = self.encoder.d_model
        self.d_know = d_know
        
        # No-exp head (content only)
        self.fc_noexp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes)
        )
        
        # Experience injection (FiLM)
        self.know_proj = nn.Linear(d_know, d_text)
        self.film_gamma = nn.Linear(d_text, d_text)
        self.film_beta = nn.Linear(d_text, d_text)
        
        # Exp head
        self.fc_exp = nn.Sequential(
            nn.Linear(d_text + additional_feature_dim_noexp, params["dense_units"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["dense_units"], num_classes)
        )
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(2 * d_text + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Agreement head (optional fusion method)
        self.agree_head = nn.Sequential(
            nn.Linear(4 * d_text, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.register_buffer("T0", torch.tensor(1.0))
        self.register_buffer("TE", torch.tensor(1.0))
        
        # Control flags
        self.use_adapter = True
    
    def _inject_experience(self, h, know_vec):
        k = self.know_proj(know_vec)
        gamma = self.film_gamma(k)
        beta = self.film_beta(k)
        return h * (1.0 + torch.tanh(gamma)) + beta
    
    def forward(self,
                input_ids,
                attention_mask,
                pos_feats,
                content_feats,
                know_vec,
                fusion="gate",
                beta=0.5,
                knowledge_dropout=0.3,
                train_mode=True):
        """
        input_ids: [B, L] LLM token ids
        attention_mask: [B, L]
        pos_feats: [B, L, POS_DIM]
        content_feats: [B, F_content]
        know_vec: [B, D_know]
        """
        # Set adapter state
        self.encoder.use_adapter = self.use_adapter
        
        # Encode with LLM + POS adapter
        h, adapter_gate = self.encoder(input_ids, attention_mask, pos_feats)
        
        # No-exp head
        z0 = self.fc_noexp(torch.cat([h, content_feats], dim=1))
        p0 = torch.softmax(z0 / self.T0, dim=-1)
        
        # Knowledge dropout during training
        if train_mode and knowledge_dropout > 0.0:
            mask = (torch.rand(know_vec.size(0), device=know_vec.device) > knowledge_dropout)
            mask = mask.float().unsqueeze(1)
            know_vec = know_vec * mask
        
        # Exp head with FiLM injection
        hE = self._inject_experience(h, know_vec)
        zE = self.fc_exp(torch.cat([hE, content_feats], dim=1))
        pE = torch.softmax(zE / self.TE, dim=-1)
        
        # Fusion
        g_val = None
        if fusion == "logit_avg":
            zf = beta * z0 + (1.0 - beta) * zE
        elif fusion == "poe":
            eps = 1e-9
            poe = (p0.clamp_min(eps) * pE.clamp_min(eps))
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
            "adapter_gate": adapter_gate
        }

class LLMNewsDataset(Dataset):
    """Dataset for LLM-based model with pre-tokenized inputs."""
    def __init__(self, input_ids, attention_masks, pos_feats, 
                 content_features, knowledge_features, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.pos_feats = pos_feats
        self.content_features = torch.tensor(content_features, dtype=torch.float)
        self.knowledge_features = torch.tensor(knowledge_features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.pos_feats[idx],
            self.content_features[idx],
            self.knowledge_features[idx],
            self.labels[idx]
        )

def prepare_llm_features(texts, labels, tokenizer, nlp, kb_embeddings, 
                         max_len=256, tfidf_vectorizer=None, verb_vectorizer=None,
                         fit_vectorizers=False):
    """
    Prepare features for LLM-based model.
    """
    # Align POS to subword tokens
    print("Aligning POS tags to subword tokens...")
    input_ids, attention_masks, pos_feats = align_pos_to_subwords(
        texts, tokenizer, nlp, max_len=max_len
    )
    
    # TF-IDF features
    if fit_vectorizers or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=150)
        tfidf_statement = tfidf_vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_statement = tfidf_vectorizer.transform(texts).toarray()
    
    # Verb TF-IDF
    verbs_corpus = []
    for text in texts:
        doc = nlp(text)
        verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
        verbs_corpus.append(" ".join(verbs) if verbs else "no_verb")
    
    if fit_vectorizers or verb_vectorizer is None:
        verb_vectorizer = TfidfVectorizer(max_features=75)
        tfidf_verbs = verb_vectorizer.fit_transform(verbs_corpus).toarray()
    else:
        tfidf_verbs = verb_vectorizer.transform(verbs_corpus).toarray()
    
    content_features = np.hstack([tfidf_statement, tfidf_verbs])
    
    # Knowledge features
    knowledge_features = link_to_external_knowledge(
        texts, kb_embeddings=kb_embeddings, top_k=3
    )
    
    print(f"Input IDs: {input_ids.shape} | POS feats: {pos_feats.shape} | "
          f"Content: {content_features.shape} | Knowledge: {knowledge_features.shape}")
    
    return (input_ids, attention_masks, pos_feats, content_features, 
            knowledge_features, tfidf_vectorizer, verb_vectorizer)

def train_llm_model(train_loader, val_loader, model, criterion, optimizer, config):
    """Training loop for LLM-based model."""
    best_val_loss = float("inf")
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler("cuda")
    fusion = config.get("fusion", "gate")
    beta = config.get("beta", 0.5)
    know_drop = config.get("knowledge_dropout", 0.3)
    lambda_agree = config.get("lambda_agree", 0.1)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids, attn_mask, pos_feats, cont, know, labels = batch
            input_ids = input_ids.to(config["device"])
            attn_mask = attn_mask.to(config["device"])
            pos_feats = pos_feats.to(config["device"])
            cont = cont.to(config["device"])
            know = know.to(config["device"])
            labels = labels.to(config["device"])
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda"):
                out = model(
                    input_ids, attn_mask, pos_feats, cont, know,
                    fusion=fusion, beta=beta,
                    knowledge_dropout=know_drop, train_mode=True
                )
                
                loss = criterion(out["z_fused"], labels) \
                       + 0.5 * criterion(out["z0"], labels) \
                       + 0.5 * criterion(out["zE"], labels)
                
                # Agreement regularization
                p0 = out["p0"].clamp_min(1e-9)
                pE = out["pE"].clamp_min(1e-9)
                symkl = 0.5 * ((p0 * (p0 / pE).log()).sum(-1) +
                               (pE * (pE / p0).log()).sum(-1)).mean()
                loss = loss + lambda_agree * symkl
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += float(loss.detach().cpu())
            running_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
            running_total += labels.size(0)
            
            pbar.set_postfix({"loss": epoch_loss / (running_total / labels.size(0))})
        
        train_epoch_loss = epoch_loss / max(len(train_loader), 1)
        train_epoch_acc = running_correct / max(running_total, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attn_mask, pos_feats, cont, know, labels = batch
                input_ids = input_ids.to(config["device"])
                attn_mask = attn_mask.to(config["device"])
                pos_feats = pos_feats.to(config["device"])
                cont = cont.to(config["device"])
                know = know.to(config["device"])
                labels = labels.to(config["device"])
                
                out = model(
                    input_ids, attn_mask, pos_feats, cont, know,
                    fusion=fusion, beta=beta,
                    knowledge_dropout=0.0, train_mode=False
                )
                
                val_loss += float(criterion(out["z_fused"], labels).cpu())
                val_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
                val_total += labels.size(0)
        
        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc*100:.2f}%")
        print(f"         | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # FIX: Robust saving with memory management
            save_path = config["best_model_path"]
            temp_path = save_path + ".tmp"
            
            # Clear CUDA cache before saving
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            try:
                # Move model to CPU to avoid CUDA memory issues during save
                state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                
                # Save to temp file with explicit flushing
                with open(temp_path, 'wb') as f:
                    torch.save(state_dict_cpu, f)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Verify temp file size
                temp_size = os.path.getsize(temp_path)
                print(f"   Temp file size: {temp_size / (1024**2):.2f} MB")
                
                if temp_size < 1_000_000:  # Less than 1 MB is suspicious
                    raise RuntimeError(f"Saved file too small ({temp_size} bytes)")
                
                # Atomic move
                import shutil
                shutil.move(temp_path, save_path)
                
                final_size = os.path.getsize(save_path)
                print(f"✅ Saved best model to {save_path} ({final_size / (1024**2):.2f} MB)")
                
                # Clean up
                del state_dict_cpu
                
            except Exception as e:
                print(f"❌ Failed to save model: {e}")
                import traceback
                traceback.print_exc()
                
                # Remove corrupt temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Fallback: save to /tmp with timestamp
                try:
                    import time
                    timestamp = int(time.time())
                    backup_path = f"/tmp/best_llm_model_epoch{epoch}_{timestamp}.pth"
                    
                    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(state_dict_cpu, backup_path)
                    
                    backup_size = os.path.getsize(backup_path)
                    print(f"⚠️  Saved backup to {backup_path} ({backup_size / (1024**2):.2f} MB)")
                    del state_dict_cpu
                    
                except Exception as e2:
                    print(f"❌ Backup save also failed: {e2}")
            
            finally:
                # Restore CUDA memory
                gc.collect()
                torch.cuda.empty_cache()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print("Early stopping.")
                break
    
    return history

if __name__ == "__main__":
    
    # Check if running LLM version
    USE_LLM = True  # Set to True to use LLM backbone
    
    if USE_LLM:
        print("\n" + "="*60)
        print("RUNNING LLM VERSION WITH POS ADAPTER")
        print("="*60 + "\n")
        
        seed = 0
        set_global_seed(seed)
        
        # Load data
        data = pd.read_csv("ReCOVery.csv").dropna(subset=["statement"])
        
        # Split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            data["statement"].tolist(), data["label"].tolist(), 
            test_size=0.1, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=seed
        )
        
        # KB embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            kb_embeddings = sentence_model.encode(
                X_train, convert_to_tensor=True, device=device
            )
            if kb_embeddings.dim() == 1:
                kb_embeddings = kb_embeddings.unsqueeze(0)
            kb_embeddings = F.normalize(kb_embeddings, p=2, dim=1)
        
        # Initialize LLM tokenizer
        LLM_NAME = "meta-llama/Llama-2-7b-hf"  # or "roberta-base" "microsoft/deberta-v3-base" "meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

        # FIX: Add padding token for LLaMA (MUST be before using tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Prepare features
        print("\nPreparing training features...")
        (ids_tr, mask_tr, pos_tr, cont_tr, know_tr, 
         tfidf_vec, verb_vec) = prepare_llm_features(
            X_train, y_train, tokenizer, nlp, kb_embeddings,
            max_len=256, fit_vectorizers=True
        )
        
        print("\nPreparing validation features...")
        (ids_va, mask_va, pos_va, cont_va, know_va, _, _) = prepare_llm_features(
            X_val, y_val, tokenizer, nlp, kb_embeddings,
            max_len=256, tfidf_vectorizer=tfidf_vec, 
            verb_vectorizer=verb_vec, fit_vectorizers=False
        )
        
        print("\nPreparing test features...")
        (ids_te, mask_te, pos_te, cont_te, know_te, _, _) = prepare_llm_features(
            X_test, y_test, tokenizer, nlp, kb_embeddings,
            max_len=256, tfidf_vectorizer=tfidf_vec,
            verb_vectorizer=verb_vec, fit_vectorizers=False
        )
        
        # Encode labels
        label_encoder = LabelEncoder().fit(y_train)
        y_train_enc = label_encoder.transform(y_train)
        y_val_enc = label_encoder.transform(y_val)
        y_test_enc = label_encoder.transform(y_test)
        
        # Class weights
        num_classes = len(label_encoder.classes_)
        counts = np.bincount(y_train_enc, minlength=num_classes)
        counts = np.where(counts == 0, 1, counts)
        weights = len(y_train_enc) / (num_classes * counts)
        print(f"Class weights: {weights.tolist()}")
        
        # Datasets
        train_dataset = LLMNewsDataset(
            ids_tr, mask_tr, pos_tr, cont_tr, know_tr, y_train_enc
        )
        val_dataset = LLMNewsDataset(
            ids_va, mask_va, pos_va, cont_va, know_va, y_val_enc
        )
        test_dataset = LLMNewsDataset(
            ids_te, mask_te, pos_te, cont_te, know_te, y_test_enc
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Config
        config = {
            "device": device,
            "num_epochs": 20,
            "patience": 5,
            "best_model_path": "best_llm_model.pth",
            "fusion": "gate",
            "beta": 0.5,
            "knowledge_dropout": 0.3,
            "lambda_agree": 0.05
        }
        config["class_weights"] = torch.tensor(weights, dtype=torch.float, device=device)
        
        # Model params
        params = {
            "dense_units": 256,
            "dropout": 0.1,
            "adapter_hidden": 64
        }
        
        # Build model
        model = L3BTwoBrainLLM(
            llm_name=LLM_NAME,
            num_classes=num_classes,
            params=params,
            additional_feature_dim_noexp=cont_tr.shape[1],
            d_know=know_tr.shape[1],
            freeze_llm=True
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        criterion = nn.CrossEntropyLoss(weight=config["class_weights"])
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-4,
            weight_decay=0.01
        )
        
        # Train
        print("\n" + "="*40)
        print("TRAINING")
        print("="*40)
        history = train_llm_model(
            train_loader, val_loader, model, criterion, optimizer, config
        )

        # Test
        print("\n" + "="*40)
        print("TESTING")
        print("="*40)
        model.load_state_dict(torch.load(config["best_model_path"], map_location=device))
        model.eval()
        
        all_labels, y0, yE, yF = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attn_mask, pos_feats, cont, know, labels = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                pos_feats = pos_feats.to(device)
                cont = cont.to(device)
                know = know.to(device)
                
                out = model(
                    input_ids, attn_mask, pos_feats, cont, know,
                    fusion="gate", knowledge_dropout=0.0, train_mode=False
                )
                
                all_labels.extend(labels.numpy())
                y0.extend(out["p0"].argmax(-1).cpu().numpy())
                yE.extend(out["pE"].argmax(-1).cpu().numpy())
                yF.extend(out["p_fused"].argmax(-1).cpu().numpy())
        
        y_true = np.array(all_labels)
        
        print("\n=== LLM Two-Brain Results ===")
        for name, preds in [("No-Exp", y0), ("Exp", yE), ("Fused", yF)]:
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average="weighted")
            print(f"{name}: Acc={acc*100:.2f}% | F1={f1*100:.2f}%")
            print(classification_report(y_true, preds, digits=4))

        print("\n✅ LLM training complete!")