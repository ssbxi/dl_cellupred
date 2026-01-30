# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LaccaseModel
from dataset import TrainingDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_recall_curve, auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_positive_data', type=str)
    parser.add_argument('--train_negative_data', type=str)
    parser.add_argument('--var_positive_data',type=str)
    parser.add_argument('--var_negative_data',type=str)
    parser.add_argument('--test_positive_data', type=str)
    parser.add_argument('--test_negative_data', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--last_layers', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=".")
    parser.add_argument('--patience', type=int, default=None, help='Number of epochs to wait before early stop.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    train_positive_data = args.train_positive_data
    train_negative_data = args.train_negative_data
    var_positive_data = args.var_positive_data
    var_negative_data = args.var_negative_data
    test_positive_data = args.test_positive_data
    test_negative_data = args.test_negative_data
    model_path = args.model_path
    BATCH_SIZE = int(args.batch_size)
    EPOCH = int(args.epoch)
    last_layers = int(args.last_layers)
    total_save_path = args.save_path
    os.makedirs(total_save_path,exist_ok=True)
    
    patience = args.patience

    # ==================== Model Loading ====================
    print("Loading model...")
    model = LaccaseModel(model_path)
    
    for name, param in model.named_parameters():
        param.requires_grad = False
        for last_layer in range(1, last_layers + 1):
            if f"encoder.layer.{model.layers - last_layer}." in name:
                param.requires_grad = True
        if "dnn" in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction="none")  
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)


    
    
    train_dataset = TrainingDataset(positive_path=train_positive_data,   
                                    negative_path=train_negative_data,   
                                    dynamic_negative_sampling=True       
                                   )
    
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True,          
                                  collate_fn=train_dataset.collate_fn,
                                  drop_last=False,       
                                  pin_memory=True,        
                                  #num_workers=16
                                 )      
    # val data
    var_dataset = TrainingDataset(positive_path=var_positive_data,
                                  negative_path=var_negative_data
    )
    var_dataloader = DataLoader(dataset=var_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=var_dataset.collate_fn,
                                drop_last=False,
                                pin_memory=True,
                                #num_workers=16
                               ) 
    
 
    epoch_losses = []
    val_history = []
   
    best_val_mcc = -1.0
    best_epoch = -1
    wait_counter = 0
    try:
        for epoch in range(EPOCH):
            # ==================== Train ====================
            model.train()
            total_loss = 0.0
            total_samples = 0
            print(f"\nEpoch {epoch+1}/{EPOCH} | Training batches: {len(train_dataloader)}")
            
            for content, label in tqdm(train_dataloader,desc=f"Training epoch {epoch+1}"):
              
                label = label.to(device)
               
                last_result = model(content)
                
                loss = criterion(last_result, label)
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.sum().item()        
                total_samples += label.size(0)          
                
            avg_train_loss = total_loss / total_samples
            epoch_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f}")
        
            
        
           
            # ==================== Validation ====================
            if ((epoch + 1) % 5 == 0) or (epoch == EPOCH - 1):
                model.eval()
                all_preds,all_probs,all_labels = [], [], []
                with torch.no_grad():
                    for data_val, label_val in tqdm(var_dataloader, desc=f"Validating epoch {epoch+1}"):
                        label_val = label_val.to(device)
                        
                        outputs = model(data_val) 
                        probs = torch.softmax(outputs, dim=1)[:, 1]  
                        preds = torch.argmax(outputs, dim=1) 
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(label_val.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
            
                val_acc = accuracy_score(all_labels, all_preds)
                val_prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
                val_rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
                val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
                val_roc_auc = roc_auc_score(all_labels, all_probs)
                val_prc_auc = average_precision_score(all_labels, all_probs)
                val_mcc = matthews_corrcoef(all_labels, all_preds)
                cm = confusion_matrix(all_labels, all_preds,labels=[1, 0])
    
                val_history.append({
                            'epoch': epoch+1,
                            'train_loss': avg_train_loss,
                            'val_acc': val_acc,
                            'val_prec': val_prec,
                            'val_rec': val_rec,
                            'val_f1': val_f1,
                            'val_roc_auc': val_roc_auc,
                            'val_prc_auc': val_prc_auc,
                            'val_mcc': val_mcc
                })
              
                print(f"\nValidation Results (Epoch {epoch+1}):")
                print(f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Prec: {val_prec:.4f} | "
                      f"Rec: {val_rec:.4f} | "
                      f"F1: {val_f1:.4f} | "
                      f"ROC-AUC: {val_roc_auc:.4f} | "
                      f"PRC-AUC: {val_prc_auc:.4f} | "
                      f"MCC: {val_mcc:.4f}")
                print("Confusion Matrix:\n", cm)
            
                # ==================== Early Stop ====================
                if val_mcc > best_val_mcc:
                    best_val_mcc = val_mcc
                    best_epoch = epoch
                    save_path = f"{total_save_path}/best_model_lastlayer{last_layers}"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_path, f"best_model_epoch{best_epoch+1}.pth"))
                    print(f"Validation MCC improved to {best_val_mcc:.4f} at epoch {best_epoch+1}")
                    wait_counter = 0
                else:
                    wait_counter += 1
                    print(f"No improvement. Patience: {wait_counter}/{patience}")
            
                # --- 只有启用早停时才提前break ---
                if patience is not None and wait_counter >= patience:
                    print(f"\nEarly stopping triggered! Best Val MCC {best_val_mcc:.4f} at epoch {best_epoch+1}")
                    break
                        
                        
    finally:
        if epoch_losses:
            #===================plt===========================
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='.', label='Train Loss (per epoch)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss vs. Epoch')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            plt.savefig(f"{total_save_path}/loss_vs_epoch_{last_layers}.pdf")   
           
            df = pd.DataFrame({'epoch': range(1, len(epoch_losses) + 1), 'loss': epoch_losses})
            df.to_csv(f'{total_save_path}/epoch_losses_{last_layers}.csv', index=False)
        if val_history:
            pd.DataFrame(val_history).to_csv(f'{total_save_path}/val_metrics_{last_layers}.csv', index=False)
    
    
    