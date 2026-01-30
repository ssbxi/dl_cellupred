#!/usr/bin/env python3

import os
import re
import copy
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from Bio import SeqIO

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score
)

from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack

from transformers import (
    T5PreTrainedModel,
    T5EncoderModel,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig
)

from transformers.modeling_outputs import SequenceClassifierOutput

from peft import (
    LoraConfig,
    inject_adapter_in_model,
    get_peft_model,
    prepare_model_for_kbit_training
)


def load_fasta(file_path):
    """加载FASTA为DataFrame，包含 seq_id 和 sequence"""
    with open(os.path.abspath(file_path)) as fasta_file:
        seq_ids, sequences = [], []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(seq_record.seq))
            seq_ids.append(seq_record.id)

    data = pd.DataFrame({"seq_id": seq_ids, "sequence": sequences})
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def preprocess_sequences(data):
    """
    空格分隔 + 非标准氨基酸替换
    """
    def _proc(seq):
        spaced = " ".join(seq)
        return re.sub(r"[UZOB]", "X", spaced)

    data["Processed"] = data["sequence"].apply(_proc)
    return data

def prepare_dataset_with_preprocessing(amp_fasta, non_amp_fasta):
    amp_df = load_fasta(amp_fasta)
    non_df = load_fasta(non_amp_fasta)

    amp_df = preprocess_sequences(amp_df)
    non_df = preprocess_sequences(non_df)

    data = pd.concat([amp_df, non_df], ignore_index=True)
    labels = [1]*len(amp_df) + [0]*len(non_df)

    return data, labels


class T5EncoderClassificationHead(nn.Module):
    """分类 Head"""
    def __init__(self, config, dropout=0.1, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class T5EncoderForSimpleSequenceClassification(T5PreTrainedModel):
   
    def __init__(self, config: T5Config, dropout=0.1, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.classifier = T5EncoderClassificationHead(config, dropout, num_labels)
        self.post_init()
        
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
    @property
    def model_parallel(self):
        return False

    @property
    def is_parallelizable(self):
        return False

    def get_encoder(self):
        return self.encoder
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = enc.last_hidden_state
        logits = self.classifier(hidden, attention_mask)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=enc.hidden_states,
                                        attentions=enc.attentions)


def load_t5_model(
    model_checkpoint:str,
    quantized:bool=True,
    lora_cfg: LoraConfig=None,
    num_labels:int=2
):
    
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, do_lower_case=False)

    if quantized:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        backbone = T5EncoderModel.from_pretrained(
            model_checkpoint,
            quantization_config=quant_cfg,
            device_map="auto"
        )
    else:
        backbone = T5EncoderModel.from_pretrained(model_checkpoint)

    model = T5EncoderForSimpleSequenceClassification(
        backbone.config, dropout=0.1, num_labels=num_labels
    )
    

    model.shared = backbone.shared
    model.encoder = backbone.encoder
    del backbone

   
    for module in model.modules():
        if "4bit" in str(type(module)) or "8bit" in str(type(module)):
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            print("gradient_checkpointing_enable:True")
            break
    
    if lora_cfg is not None:
        
        model = inject_adapter_in_model(lora_cfg,model)
        print("get_peft_model:True")
    
     
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model, tokenizer


def create_hf_dataset(tokenizer, df_data, labels, max_length=1024):
    seq_list = df_data["Processed"].tolist()
    tokenized = tokenizer(
        seq_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )
    return Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    mcc = matthews_corrcoef(labels, preds)

    try:
        auc = roc_auc_score(labels, logits[:,1])
    except:
        auc = float("nan")
        
    try:
        pr_auc = average_precision_score(labels, logits[:, 1])
    except:
        pr_auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": auc,
        "prc_auc": pr_auc 
    }



def train_t5_model(args):
    """主训练函数"""

   
    set_seed(args.seed)

    train_df, train_labels = prepare_dataset_with_preprocessing(
        args.train_amp_fasta, args.train_non_amp_fasta
    )
    val_df, val_labels = prepare_dataset_with_preprocessing(
        args.val_amp_fasta, args.val_non_amp_fasta
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="lora_only",
        target_modules=["q","k","v"],
        lora_dropout=0.1
    )

    model, tokenizer = load_t5_model(
        args.model_checkpoint,
        quantized=args.quantized,
        lora_cfg=lora_config
    )

    train_ds = create_hf_dataset(tokenizer, train_df, train_labels)
    val_ds = create_hf_dataset(tokenizer, val_df, val_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        save_safetensors=False,
        fp16=True,
        weight_decay=0.1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n=== Training Start ===")
    trainer.train()

    # Test evaluation
    test_df, test_labels = prepare_dataset_with_preprocessing(
        args.test_amp_fasta, args.test_non_amp_fasta
    )
    test_ds = create_hf_dataset(tokenizer, test_df, test_labels)

    print("\n=== Test Evaluation ===")
    preds_out = trainer.predict(test_ds)
    logits = preds_out.predictions
    true_labels = preds_out.label_ids

    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    test_metrics = compute_metrics((logits, true_labels))

    
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(trainer.state.log_history).to_csv(f"{args.output_dir}/train_log.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(f"{args.output_dir}/test_metrics.csv", index=False)
    np.save(os.path.join(args.output_dir, "test_probabilities.npy"), probs)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), true_labels)

    print("\nSaved results in:", args.output_dir)
    print("Test metrics:", test_metrics)



def main():
    parser = argparse.ArgumentParser(description="Train T5 AMP Classifier")

    parser.add_argument("--train_amp_fasta", type=str, required=True)
    parser.add_argument("--train_non_amp_fasta", type=str, required=True)
    parser.add_argument("--val_amp_fasta", type=str, required=True)
    parser.add_argument("--val_non_amp_fasta", type=str, required=True)
    parser.add_argument("--test_amp_fasta", type=str, required=True)
    parser.add_argument("--test_non_amp_fasta", type=str, required=True)

    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="路径到 ProtT5 checkpoint")
    parser.add_argument("--output_dir", type=str, default="t5_output")

    parser.add_argument("--quantized", action="store_true",
                        help="是否使用 4bit 量化")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=4)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_t5_model(args)

if __name__ == "__main__":
    main()


