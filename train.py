import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.dyg_stmt import DyGSTMTNet
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics, print_metrics_summary
from dataset.nthu_data_loader import ConsolidatedNTHUDataset

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch=1):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_batches = len(dataloader)
    
    for b_idx, batch in enumerate(dataloader):
        eye_b, mouth_b, geo_b, cond_b, y_drowsy, y_eye, y_mouth, y_head = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(eye_b, mouth_b, geo_b, cond_b)
            targets = {
                'drowsy': y_drowsy,
                'eye': y_eye,
                'mouth': y_mouth,
                'head': y_head
            }
            loss, loss_dict = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * len(y_drowsy)
        probs = torch.sigmoid(outputs['drowsy_logits']).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y_drowsy.cpu().numpy())
        
        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == total_batches:
            batch_acc = np.mean(((np.array(all_preds[-len(y_drowsy):]) >= 0.5) == y_drowsy.cpu().numpy()))
            print(f"  [Epoch {epoch:02d}] Batch {b_idx+1:03d}/{total_batches:03d} | Batch Loss: {loss.item():.4f} | Batch Acc: {batch_acc*100:.1f}%", flush=True)
            
    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    metrics = compute_drowsiness_metrics(all_targets, all_preds)
    metrics['loss'] = epoch_loss
    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            eye_b, mouth_b, geo_b, cond_b, y_drowsy, y_eye, y_mouth, y_head = [x.to(device) for x in batch]
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(eye_b, mouth_b, geo_b, cond_b)
                targets = {
                    'drowsy': y_drowsy,
                    'eye': y_eye,
                    'mouth': y_mouth,
                    'head': y_head
                }
                loss, _ = criterion(outputs, targets)
                
            running_loss += loss.item() * len(y_drowsy)
            probs = torch.sigmoid(outputs['drowsy_logits']).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_drowsy.cpu().numpy())
            
    eval_loss = running_loss / max(1, len(dataloader.dataset))
    metrics = compute_drowsiness_metrics(all_targets, all_preds)
    metrics['loss'] = eval_loss
    return metrics, all_preds, all_targets


def train_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Training Engine] Using Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load Consolidated Memory-Mapped Datasets
    print(f"[Data] Loading train dataset from {args.train_file}...", flush=True)
    train_dataset = ConsolidatedNTHUDataset(args.train_file, augment=True)
    
    if os.path.exists(args.eval_file):
        print(f"[Data] Loading eval dataset from {args.eval_file}...", flush=True)
        val_dataset = ConsolidatedNTHUDataset(args.eval_file, augment=False)
    else:
        train_len = int(0.85 * len(train_dataset))
        val_len = len(train_dataset) - train_len
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])
        print(f"[Data] Split into {train_len} train / {val_len} val sequences.", flush=True)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))
    
    # Model, Loss, Optimizer
    model = DyGSTMTNet(num_conditions=5, feature_dim=128, gru_hidden=128, dropout=args.dropout).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.65, lambda_eye=0.25, lambda_mouth=0.20, lambda_head=0.15)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\nStarting DyG-STMT Training Loop...", flush=True)
    print(f"Hyperparameters: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, WeightDecay={args.weight_decay}, Dropout={args.dropout}\n", flush=True)
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch=epoch)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])
        
        elapsed = time.time() - t0
        
        print(f"\n>>> Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s)", flush=True)
        print(f"    Train -> Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']*100:.2f}% | F1: {train_metrics['f1_score']*100:.2f}% | AUC: {train_metrics['roc_auc']*100:.2f}%", flush=True)
        print(f"    Val   -> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1_score']*100:.2f}% | AUC: {val_metrics['roc_auc']*100:.2f}%", flush=True)
              
        if val_metrics['f1_score'] > best_val_f1 or (val_metrics['f1_score'] == best_val_f1 and val_metrics['accuracy'] > best_val_acc):
            best_val_f1 = val_metrics['f1_score']
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            patience_counter = 0
            
            save_path = os.path.join(args.checkpoint_dir, 'best_dyg_stmt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, save_path)
            print(f"    ⭐ NEW BEST MODEL SAVED to {save_path} (Val F1: {best_val_f1*100:.2f}%, Val Acc: {best_val_acc*100:.2f}%)\n", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement in {args.patience} epochs. Stopping training at epoch {epoch}.", flush=True)
                break
                
    print(f"\n==================== Training Finished ====================", flush=True)
    print(f"  Best Epoch : {best_epoch}", flush=True)
    print(f"  Best Val F1: {best_val_f1*100:.2f}%", flush=True)
    print(f"  Best Val Acc: {best_val_acc*100:.2f}%", flush=True)
    print("===========================================================\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data_cache/train_consolidated.npz')
    parser.add_argument('--eval_file', type=str, default='data_cache/eval_consolidated.npz')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.30)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    
    train_pipeline(args)
