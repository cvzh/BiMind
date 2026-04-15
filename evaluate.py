import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils import summarize_gates


def test_model(test_loader, model, config: dict, feature_names=None, vocab=None):
    """Evaluate the custom-transformer (non-LLM) BiMind model on the test set."""
    device = config["device"]
    fusion = config.get("fusion", "logit_avg")
    beta = config.get("beta", 0.5)

    model.to(device)
    model.eval()

    all_labels, y0, yE, yF = [], [], [], []
    agree_vals, vox_gains, gate_vals = [], [], []

    with torch.no_grad():
        for seqs, posf, cont, know, labels in test_loader:
            seqs = seqs.to(device)
            posf = posf.to(device)
            cont = cont.to(device)
            know = know.to(device)
            labels = labels.to(device)

            out = model(
                seqs, posf, cont, know,
                fusion=fusion, beta=beta,
                knowledge_dropout=0.0, train_mode=False,
            )

            all_labels.extend(labels.cpu().numpy())
            y0.extend(out["p0"].argmax(-1).cpu().numpy())
            yE.extend(out["pE"].argmax(-1).cpu().numpy())
            yF.extend(out["p_fused"].argmax(-1).cpu().numpy())

            p0 = out["p0"].clamp_min(1e-9)
            pE = out["pE"].clamp_min(1e-9)
            skl = 0.5 * (
                (p0 * (p0 / pE).log()).sum(-1) + (pE * (pE / p0).log()).sum(-1)
            )
            agree_vals.extend(skl.cpu().numpy())

            gy = (
                out["zE"].gather(1, labels.view(-1, 1))
                - out["z0"].gather(1, labels.view(-1, 1))
            )
            vox_gains.extend(gy.squeeze(1).cpu().numpy())

            if out.get("gate") is not None:
                gate_vals.append(out["gate"].cpu().numpy())

    y_true = np.array(all_labels)
    y0 = np.array(y0)
    yE = np.array(yE)
    yF = np.array(yF)

    def _report(name: str, yhat: np.ndarray) -> None:
        acc = accuracy_score(y_true, yhat)
        f1w = f1_score(y_true, yhat, average="weighted")
        print(f"✅ {name} — Acc: {acc * 100:.2f}% | F1(w): {f1w * 100:.2f}%")
        print(classification_report(y_true, yhat, digits=4))

    print("\n=== L³B Two-Brain Evaluation ===")
    _report("No-Experience", y0)
    _report("Have-Experience", yE)
    _report("Fused", yF)

    print(
        f"Agreement (sym-KL) mean: {np.mean(agree_vals):.4f} | "
        f"median: {np.median(agree_vals):.4f}"
    )
    print(
        f"VoX gain mean: {np.mean(vox_gains):.4f} | "
        f"median: {np.median(vox_gains):.4f} | "
        f"positive%: {100 * np.mean(np.array(vox_gains) > 0):.2f}%"
    )

    if fusion == "gate" and gate_vals:
        summarize_gates(np.concatenate(gate_vals, axis=0), name="Test Gate")

    os.makedirs("reports", exist_ok=True)
    with open("reports/test_results.txt", "w") as f:
        f.write("Two-Brain Results\n")
        f.write(f"Agreement sym-KL mean: {np.mean(agree_vals):.4f}\n")
        f.write(f"VoX gain mean: {np.mean(vox_gains):.4f}\n")
