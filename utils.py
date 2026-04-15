import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import spacy
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Global singletons (loaded once at import time)
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_md")

sentence_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
).to("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# POS constants
# ---------------------------------------------------------------------------
POS2ID = {"VERB": 0, "AUX": 0, "NOUN": 1, "ADJ": 2, "ADV": 3}
POS_DIM = 5


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def tokenize(text: str):
    doc = nlp.make_doc(text)
    return [t.text.lower() for t in doc if not (t.is_space or t.is_punct)]


def build_vocab(texts, min_freq: int = 1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def text_to_sequence(text: str, vocab: dict, max_len: int):
    seq = [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)]
    return seq[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(seq))


def extract_verbs(text: str):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.pos_ == "VERB"]


# ---------------------------------------------------------------------------
# External-knowledge retrieval
# ---------------------------------------------------------------------------
def link_to_external_knowledge(
    texts, kb_embeddings: torch.Tensor, top_k: int = 5
) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    device = kb_embeddings.device
    q = sentence_model.encode(texts, convert_to_tensor=True, device=device)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = F.normalize(q, p=2, dim=1)

    scores = q @ kb_embeddings.T
    k = min(top_k, kb_embeddings.size(0))
    vals, idx = torch.topk(scores, k=k, dim=1)

    sel = kb_embeddings.index_select(0, idx.reshape(-1)).reshape(
        q.size(0), k, -1
    )
    agg = sel.mean(dim=1)

    max_sim = vals.max(dim=1).values.unsqueeze(1)
    mean_sim = vals.mean(dim=1, keepdim=True)

    out = torch.cat([agg, max_sim, mean_sim], dim=1)
    return out.detach().cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# POS helpers
# ---------------------------------------------------------------------------
def pos_onehot_from_tag(tag: str):
    v = [0] * POS_DIM
    v[POS2ID.get(tag, 4)] = 1
    return v


def pos_mats_for_texts(texts, max_len: int) -> np.ndarray:
    mats = []
    for doc in nlp.pipe(texts, batch_size=256, n_process=1):
        toks = [
            tok.pos_ for tok in doc if not (tok.is_space or tok.is_punct)
        ][:max_len]
        rows = [pos_onehot_from_tag(tag) for tag in toks]
        while len(rows) < max_len:
            rows.append(pos_onehot_from_tag("OTHER"))
        mats.append(np.asarray(rows, dtype=np.float32))
    return np.stack(mats, axis=0)


def align_pos_to_subwords(texts, tokenizer, nlp_model, max_len: int = 512):
    """Align spaCy POS tags to LLM subword tokens.

    Returns:
        input_ids       [N, L]
        attention_mask  [N, L]
        pos_feats       [N, L, POS_DIM]
    """
    all_input_ids, all_attn_masks, all_pos_feats = [], [], []

    for text in texts:
        doc = nlp_model(text)
        word_pos = [
            (tok.text, tok.pos_, tok.idx, tok.idx + len(tok.text))
            for tok in doc
            if not (tok.is_space or tok.is_punct)
        ]

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)

        char_to_pos = {}
        for _, pos, start, end in word_pos:
            for c in range(start, end):
                char_to_pos[c] = pos

        pos_ids = []
        for start, end in offsets:
            start, end = start.item(), end.item()
            if start == end:
                pos_ids.append("OTHER")
            else:
                pos_ids.append(char_to_pos.get(start, "OTHER"))

        pos_feats = torch.tensor(
            [pos_onehot_from_tag(p) for p in pos_ids], dtype=torch.float
        )

        all_input_ids.append(input_ids)
        all_attn_masks.append(attention_mask)
        all_pos_feats.append(pos_feats)

    return (
        torch.stack(all_input_ids),
        torch.stack(all_attn_masks),
        torch.stack(all_pos_feats),
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------
def summarize_gates(gates_np: np.ndarray, name: str = "Gate") -> None:
    if gates_np is None or gates_np.size == 0:
        print(f"{name}: no gate values collected.")
        return
    g = gates_np.ravel()
    print(
        f"{name} — mean: {np.mean(g):.4f} | median: {np.median(g):.4f} | "
        f"%<0.3: {100.0 * np.mean(g < 0.3):.2f}% | "
        f"%>0.7: {100.0 * np.mean(g > 0.7):.2f}%"
    )
