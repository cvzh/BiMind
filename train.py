import gc
import os
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import summarize_gates


def train_model(train_loader, val_loader, model, criterion, optimizer, config: dict):
    """Training loop for the custom-transformer (non-LLM) BiMind model."""
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

        for seqs, posf, cont, know, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}"
        ):
            seqs = seqs.to(config["device"])
            posf = posf.to(config["device"])
            cont = cont.to(config["device"])
            know = know.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(
                    seqs, posf, cont, know,
                    fusion=fusion, beta=beta,
                    knowledge_dropout=know_drop, train_mode=True,
                )
                loss = (
                    criterion(out["z_fused"], labels)
                    + 0.5 * criterion(out["z0"], labels)
                    + 0.5 * criterion(out["zE"], labels)
                )
                p0 = out["p0"].clamp_min(1e-9)
                pE = out["pE"].clamp_min(1e-9)
                symkl = 0.5 * (
                    (p0 * (p0 / pE).log()).sum(-1)
                    + (pE * (pE / p0).log()).sum(-1)
                ).mean()
                loss = loss + lambda_agree * symkl

                for layer in model.transformer_encoder.layers:
                    if hasattr(layer, "get_entropy_loss"):
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
        train_epoch_acc = running_correct / max(running_total, 1)
        print(
            f"Epoch {epoch + 1} Training Loss: {train_epoch_loss:.4f} | "
            f"Acc: {train_epoch_acc * 100:.2f}%"
        )

        # ---- Validation ----
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

                out = model(
                    seqs, posf, cont, know,
                    fusion=fusion, beta=beta,
                    knowledge_dropout=0.0, train_mode=False,
                )
                val_loss += float(criterion(out["z_fused"], labels).detach().cpu())
                val_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
                val_total += labels.size(0)

                if out.get("gate") is not None:
                    val_gates.append(out["gate"].cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"Validation Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}%")

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_acc)

        if fusion == "gate" and val_gates:
            summarize_gates(np.concatenate(val_gates, axis=0), name="Val Gate")

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


def train_llm_model(train_loader, val_loader, model, criterion, optimizer, config: dict):
    """Training loop for the LLM-backbone BiMind model."""
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
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
                    knowledge_dropout=know_drop, train_mode=True,
                )
                loss = (
                    criterion(out["z_fused"], labels)
                    + 0.5 * criterion(out["z0"], labels)
                    + 0.5 * criterion(out["zE"], labels)
                )
                p0 = out["p0"].clamp_min(1e-9)
                pE = out["pE"].clamp_min(1e-9)
                symkl = 0.5 * (
                    (p0 * (p0 / pE).log()).sum(-1)
                    + (pE * (pE / p0).log()).sum(-1)
                ).mean()
                loss = loss + lambda_agree * symkl

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            running_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
            running_total += labels.size(0)
            pbar.set_postfix({"loss": epoch_loss / max(running_total / labels.size(0), 1)})

        train_epoch_loss = epoch_loss / max(len(train_loader), 1)
        train_epoch_acc = running_correct / max(running_total, 1)

        # ---- Validation ----
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
                    knowledge_dropout=0.0, train_mode=False,
                )
                val_loss += float(criterion(out["z_fused"], labels).cpu())
                val_correct += int((out["z_fused"].argmax(-1) == labels).sum().item())
                val_total += labels.size(0)

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_epoch_loss:.4f} | "
            f"Train Acc: {train_epoch_acc * 100:.2f}%"
        )
        print(
            f"         | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            save_path = config["best_model_path"]
            temp_path = save_path + ".tmp"

            torch.cuda.empty_cache()
            gc.collect()

            try:
                state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}

                with open(temp_path, "wb") as f:
                    torch.save(state_dict_cpu, f)
                    f.flush()
                    os.fsync(f.fileno())

                temp_size = os.path.getsize(temp_path)
                print(f"   Temp file size: {temp_size / (1024 ** 2):.2f} MB")

                if temp_size < 1_000_000:
                    raise RuntimeError(f"Saved file too small ({temp_size} bytes)")

                shutil.move(temp_path, save_path)
                final_size = os.path.getsize(save_path)
                print(
                    f"✅ Saved best model to {save_path} "
                    f"({final_size / (1024 ** 2):.2f} MB)"
                )
                del state_dict_cpu

            except Exception as e:
                import traceback

                print(f"❌ Failed to save model: {e}")
                traceback.print_exc()

                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

                try:
                    timestamp = int(time.time())
                    backup_path = (
                        f"/tmp/best_llm_model_epoch{epoch}_{timestamp}.pth"
                    )
                    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(state_dict_cpu, backup_path)
                    backup_size = os.path.getsize(backup_path)
                    print(
                        f"⚠️  Saved backup to {backup_path} "
                        f"({backup_size / (1024 ** 2):.2f} MB)"
                    )
                    del state_dict_cpu
                except Exception as e2:
                    print(f"❌ Backup save also failed: {e2}")

            finally:
                gc.collect()
                torch.cuda.empty_cache()

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print("Early stopping.")
                break

    return history
