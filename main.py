"""
main.py — Entry point for BiMind LLM fake-news detector.

Run:
    python main.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import LLMNewsDataset
from features import prepare_llm_features
from models import L3BTwoBrainLLM
from train import train_llm_model
from utils import nlp, sentence_model, set_global_seed


def main() -> None:
    print("\n" + "=" * 60)
    print("RUNNING LLM VERSION WITH POS ADAPTER")
    print("=" * 60 + "\n")

    seed = 0
    set_global_seed(seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = pd.read_csv("ReCOVery.csv").dropna(subset=["statement"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data["statement"].tolist(),
        data["label"].tolist(),
        test_size=0.1,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=seed
    )

    # ------------------------------------------------------------------
    # Knowledge base (sentence embeddings of training texts)
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        kb_embeddings = sentence_model.encode(
            X_train, convert_to_tensor=True, device=device
        )
        if kb_embeddings.dim() == 1:
            kb_embeddings = kb_embeddings.unsqueeze(0)
        kb_embeddings = F.normalize(kb_embeddings, p=2, dim=1)

    # ------------------------------------------------------------------
    # LLM tokenizer
    # ------------------------------------------------------------------
    LLM_NAME = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    print("\nPreparing training features...")
    (ids_tr, mask_tr, pos_tr, cont_tr, know_tr, tfidf_vec, verb_vec) = (
        prepare_llm_features(
            X_train, y_train, tokenizer, nlp, kb_embeddings,
            max_len=256, fit_vectorizers=True,
        )
    )

    print("\nPreparing validation features...")
    (ids_va, mask_va, pos_va, cont_va, know_va, _, _) = prepare_llm_features(
        X_val, y_val, tokenizer, nlp, kb_embeddings,
        max_len=256,
        tfidf_vectorizer=tfidf_vec,
        verb_vectorizer=verb_vec,
        fit_vectorizers=False,
    )

    print("\nPreparing test features...")
    (ids_te, mask_te, pos_te, cont_te, know_te, _, _) = prepare_llm_features(
        X_test, y_test, tokenizer, nlp, kb_embeddings,
        max_len=256,
        tfidf_vectorizer=tfidf_vec,
        verb_vectorizer=verb_vec,
        fit_vectorizers=False,
    )

    # ------------------------------------------------------------------
    # Label encoding & class weights
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder().fit(y_train)
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    counts = np.bincount(y_train_enc, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = len(y_train_enc) / (num_classes * counts)
    print(f"Class weights: {weights.tolist()}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_dataset = LLMNewsDataset(ids_tr, mask_tr, pos_tr, cont_tr, know_tr, y_train_enc)
    val_dataset = LLMNewsDataset(ids_va, mask_va, pos_va, cont_va, know_va, y_val_enc)
    test_dataset = LLMNewsDataset(ids_te, mask_te, pos_te, cont_te, know_te, y_test_enc)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # ------------------------------------------------------------------
    # Config & model
    # ------------------------------------------------------------------
    config = {
        "device": device,
        "num_epochs": 20,
        "patience": 5,
        "best_model_path": "best_llm_model.pth",
        "fusion": "gate",
        "beta": 0.5,
        "knowledge_dropout": 0.3,
        "lambda_agree": 0.05,
    }
    config["class_weights"] = torch.tensor(weights, dtype=torch.float, device=device)

    params = {"dense_units": 256, "dropout": 0.1, "adapter_hidden": 64}

    model = L3BTwoBrainLLM(
        llm_name=LLM_NAME,
        num_classes=num_classes,
        params=params,
        additional_feature_dim_noexp=cont_tr.shape[1],
        d_know=know_tr.shape[1],
        freeze_llm=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=config["class_weights"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        weight_decay=0.01,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("TRAINING")
    print("=" * 40)
    train_llm_model(train_loader, val_loader, model, criterion, optimizer, config)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("TESTING")
    print("=" * 40)

    model.load_state_dict(
        torch.load(config["best_model_path"], map_location=device)
    )
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
                fusion="gate", knowledge_dropout=0.0, train_mode=False,
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
        print(f"{name}: Acc={acc * 100:.2f}% | F1={f1 * 100:.2f}%")
        print(classification_report(y_true, preds, digits=4))

    print("\n✅ LLM training complete!")


if __name__ == "__main__":
    main()
