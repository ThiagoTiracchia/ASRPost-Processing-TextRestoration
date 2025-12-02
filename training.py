
import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

def train_one_epoch(model, device, data_loader, criterion_initial, criterion_final, criterion_capitalization, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    punct_ini_pred, punct_ini_ground_truth = [], []
    punct_fin_pred, punct_fin_ground_truth = [], []
    cap_pred, cap_ground_truth = [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for embedding_group, punt_inicial_group, punt_final_group, capitalizacion_group in tqdm(data_loader): # train data loader (un batch)
        batch_loss = 0.0
        batch_samples = 0
        for embedding, true_punt_inicial_idx, true_punt_final_idx, true_capitalizacion_idx in zip(embedding_group, punt_inicial_group, punt_final_group, capitalizacion_group):
            total_samples += 1
            batch_samples += 1
            # Para cada "oración" (instancia) en el batch
            instance_length = embedding.size(0)
            embedding = embedding.to(device).unsqueeze(0)

            true_punt_inicial_idx = true_punt_inicial_idx.to(device)
            true_punt_final_idx = true_punt_final_idx.to(device)
            true_capitalizacion_idx = true_capitalizacion_idx.to(device)

            out_punct_ini, out_punct_fin, out_cap = model(embedding) # forward pass - predicciones de las puntuaciones y capitalización de la instacia
            out_punct_ini = out_punct_ini.permute(0, 2, 1)
            out_punct_fin = out_punct_fin.permute(0, 2, 1)
            out_cap = out_cap.permute(0, 2, 1)

            batch_loss += criterion_initial(out_punct_ini, true_punt_inicial_idx)
            batch_loss += criterion_final(out_punct_fin, true_punt_final_idx)
            batch_loss += criterion_capitalization(out_cap, true_capitalizacion_idx)
            batch_loss /= 3
            _, pred_punct_ini = torch.max(out_punct_ini, dim=1)
            _, pred_punct_fin = torch.max(out_punct_fin, dim=1)
            _, pred_cap = torch.max(out_cap, dim=1)

            correct_punct_ini = (pred_punct_ini == true_punt_inicial_idx)
            correct_punct_fin = (pred_punct_fin == true_punt_final_idx)
            correct_cap = (pred_cap == true_capitalizacion_idx)
            punct_ini_pred.extend(pred_punct_ini.cpu().squeeze(0).tolist())
            punct_ini_ground_truth.extend(true_punt_inicial_idx.cpu().squeeze(0).tolist())
            punct_fin_pred.extend(pred_punct_fin.cpu().squeeze(0).tolist())
            punct_fin_ground_truth.extend(true_punt_final_idx.cpu().squeeze(0).tolist())
            cap_pred.extend(pred_cap.cpu().squeeze(0).tolist())
            cap_ground_truth.extend(true_capitalizacion_idx.cpu().squeeze(0).tolist())

            total_correct += torch.sum(correct_punct_ini + correct_punct_fin + correct_cap)

        total_loss += batch_loss.item()
        batch_loss /= batch_samples
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total_samples

    return (
        avg_loss, f1_score(punct_ini_pred, punct_ini_ground_truth, zero_division=0, average="macro"),
        f1_score(punct_fin_pred, punct_fin_ground_truth, average="macro", labels=[0,1,2,3], zero_division=0),
        f1_score(cap_pred, cap_ground_truth, average="macro", labels=[0,1,2,3], zero_division=0)
    )

def validation(model, device, data_loader, criterion_initial, criterion_final, criterion_capitalization, optimizer):
    model.eval()
    running_loss, correct, total_samples = 0.0, 0, 0
    punct_ini_pred, punct_ini_ground_truth = [], []
    punct_fin_pred, punct_fin_ground_truth = [], []
    cap_pred, cap_ground_truth = [], []
    total_samples = 0
    with torch.no_grad():
        for embedding_group, punt_inicial_group, punt_final_group, capitalizacion_group in data_loader:
            for embedding, true_punt_inicial_idx, true_punt_final_idx, true_capitalizacion_idx in zip(embedding_group,
                                                                                                      punt_inicial_group,
                                                                                                      punt_final_group,
                                                                                                      capitalizacion_group):
                # Para cada "oración" (instancia) en el batch
                total_samples += 1
                instance_length = embedding.size(0)
                embedding = embedding.to(device).unsqueeze(0)

                true_punt_inicial_idx = true_punt_inicial_idx.to(device)
                true_punt_final_idx = true_punt_final_idx.to(device)
                true_capitalizacion_idx = true_capitalizacion_idx.to(device)

                out_punct_ini, out_punct_fin, out_cap = model(embedding)  # forward pass - predicciones de las puntuaciones y capitalización de la instacia
                out_punct_ini = out_punct_ini.permute(0, 2, 1)
                out_punct_fin = out_punct_fin.permute(0, 2, 1)
                out_cap = out_cap.permute(0, 2, 1)

                running_loss += criterion_initial(out_punct_ini, true_punt_inicial_idx)
                running_loss += criterion_final(out_punct_fin, true_punt_final_idx)
                running_loss += criterion_capitalization(out_cap, true_capitalizacion_idx)

                _, pred_punct_ini = torch.max(out_punct_ini, dim=1)
                _, pred_punct_fin = torch.max(out_punct_fin, dim=1)
                _, pred_cap = torch.max(out_cap, dim=1)

                punct_ini_pred.extend(pred_punct_ini.cpu().squeeze(0).tolist())
                punct_ini_ground_truth.extend(true_punt_inicial_idx.cpu().squeeze(0).tolist())
                punct_fin_pred.extend(pred_punct_fin.cpu().squeeze(0).tolist())
                punct_fin_ground_truth.extend(true_punt_final_idx.cpu().squeeze(0).tolist())
                cap_pred.extend(pred_cap.cpu().squeeze(0).tolist())
                cap_ground_truth.extend(true_capitalizacion_idx.cpu().squeeze(0).tolist())

    avg_loss = running_loss / total_samples
    return (
        avg_loss, f1_score(punct_ini_pred, punct_ini_ground_truth, zero_division=0, average="macro"),
        f1_score(punct_fin_pred, punct_fin_ground_truth, average="macro", labels=[0,1,2,3], zero_division=0),
        f1_score(cap_pred, cap_ground_truth, average="macro", labels=[0,1,2,3], zero_division=0)
    )

def training_loop(model, device, train_loader, val_loader, criterion_initial, criterion_final, criterion_capitalization, optimizer, num_epochs=5, name_prefix="model"):
    with open(f"{name_prefix}_training_log.txt", "w") as log_file:
        log_file.write("Epoch,Train Loss,Val Loss,F1 Punct Ini Train,F1 Punct Ini Val,F1 Punct Fin Train,F1 Punct Fin Val,F1 Cap Train,F1 Cap Val\n")
        best_val_loss = float("inf")
        best_val_f1_punct_ini = 0.0
        best_val_f1_punct_fin = 0.0
        best_val_f1_cap = 0.0
        history = {"train_loss": [], "train_acc": [], "f1_punct_ini_train": [], "f1_punct_final_train": [], "f1_cap_train": [],
                   "val_loss": [], "val_acc": [], "f1_punct_ini_val": [], "f1_punct_final_val": [], "f1_cap_val": []}

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, f1_punct_ini_train, f1_punct_final_train, f1_cap_train = train_one_epoch(
                model, device, train_loader, criterion_initial, criterion_final, criterion_capitalization, optimizer
            )
            val_loss, f1_punct_ini_val, f1_punct_final_val, f1_cap_val = validation(
                model, device, val_loader, criterion_initial, criterion_final, criterion_capitalization, optimizer
            )

            history["train_loss"].append(train_loss)
            history["f1_punct_ini_train"].append(f1_punct_ini_train)
            history["f1_punct_final_train"].append(f1_punct_final_train)
            history["f1_cap_train"].append(f1_cap_train)
            history["val_loss"].append(val_loss)
            history["f1_punct_ini_val"].append(f1_punct_ini_val)
            history["f1_punct_final_val"].append(f1_punct_final_val)
            history["f1_cap_val"].append(f1_cap_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1_punct_ini = f1_punct_ini_val
                best_val_f1_punct_fin = f1_punct_final_val
                best_val_f1_cap = f1_cap_val

            print(f"Train Loss: {train_loss:.4f} | F1 Punct Ini: {f1_punct_ini_train:.4f} | F1 Punct Fin: {f1_punct_final_train:.4f} | F1 Cap: {f1_cap_train:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | F1 Punct Ini: {f1_punct_ini_val:.4f} | F1 Punct Fin: {f1_punct_final_val:.4f} | F1 Cap: {f1_cap_val:.4f}")
            log_file.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{f1_punct_ini_train:.4f},{f1_punct_ini_val:.4f},{f1_punct_final_train:.4f},{f1_punct_final_val:.4f},{f1_cap_train:.4f},{f1_cap_val:.4f}\n")
            log_file.flush()
            torch.save(model.state_dict(), f"{name_prefix}_epoch_{epoch+1}.pth")
            gc.collect()

    print(f"\n Training complete. Best Val Loss = {best_val_loss:.4f}, Best Val F1 Punct Ini = {best_val_f1_punct_ini:.4f}, Best Val F1 Punct Fin = {best_val_f1_punct_fin:.4f}, Best Val F1 Cap = {best_val_f1_cap:.4f}")
    return history

def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))

    # Loss curve
    plt.subplot(1, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.savefig("loss_curves.png")
    plt.close()

    plt.figure(figsize=(18, 5))
    plt.tight_layout()

    # Subgráfico 1
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["f1_punct_ini_train"], label="Train F1")
    plt.plot(epochs, history["f1_punct_ini_val"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 para puntación inicial")
    plt.ylim(-0.05, 1)
    plt.xlim(0, len(epochs)+1)
    plt.legend()

    # Subgráfico 2
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["f1_punct_final_train"], label="Train F1")
    plt.plot(epochs, history["f1_punct_final_val"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 para puntación final")
    plt.ylim(-0.05, 1)
    plt.xlim(0, len(epochs)+1)
    plt.legend()

    # Subgráfico 3
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["f1_cap_train"], label="Train F1")
    plt.plot(epochs, history["f1_cap_val"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 para capitalización")
    plt.ylim(-0.05, 1)
    plt.xlim(0, len(epochs)+1)
    plt.legend()

    plt.savefig("f1_curves.png")
    plt.close()