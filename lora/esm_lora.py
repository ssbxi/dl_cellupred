import torch
import torch.nn as nn
import pandas as pd
from transformers import (AutoTokenizer,
                          EsmForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          BitsAndBytesConfig)
from Bio import SeqIO
from datasets import Dataset
from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  LoftQConfig,
                  get_peft_config,
                  PeftModel,
                  PeftConfig,
                  prepare_model_for_kbit_training)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,auc
)
from transformers import AutoModelForSequenceClassification
import bitsandbytes as bnb
import gc
import wandb
import os
import argparse
import random
import warnings
warnings.filterwarnings("ignore")


def load_fasta(file_path):
    with open(os.path.abspath(file_path)) as fasta_file:
        sequences = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(seq_record.seq))
    return sequences


def prepare_dataset(amp_fasta, non_amp_fasta):
    amp_data = load_fasta(amp_fasta)
    non_amp_data = load_fasta(non_amp_fasta)
    sequences = amp_data + non_amp_data
    labels = [1]*len(amp_data) + [0]*len(non_amp_data)
    return sequences, labels


def load_esm2_model(shorthand, quantized=True):
    model_versions = {
        "8m": "/home/nky01/anlysis/biye_cwq/dl_analysis/cellose/db/esm-2/8m",
        "35m":"/home/nky01/anlysis/biye_cwq/dl_analysis/cellose/db/esm-2/35m",
        "150m":"/home/nky01/anlysis/biye_cwq/dl_analysis/cellose/db/esm-2/150m",
        "650m":"/home/nky01/anlysis/biye_cwq/dl_analysis/cellose/db/esm-2/650m",
        "3B": "/home/nky01/anlysis/biye_cwq/dl_analysis/cellose/db/esm-2/3B",
        "1b650m":"/home/nky01/anlysis/biye_cwq/dl_analysis/cellose/db/esm-1/esm-1b"
    }
    model_name = model_versions[shorthand]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantized:
        print("quantized:true")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        llm_int8_skip_modules=["classifier", "pre_classifier"],
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        
        model = EsmForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            quantization_config=bnb_config
        )
    else:
        model = EsmForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
    return model, tokenizer


def tokenize_and_create_dataset(sequences, labels, tokenizer, max_length=1022):
    tokenized = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return Dataset.from_dict({**tokenized, "labels": labels})


def compute_metrics(p):
    logits, true_labels = p
    
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels,
                                                               predictions,
                                                               average='binary')
    
    roc_auc = roc_auc_score(true_labels, logits[:, 1]) if len(np.unique(true_labels)) == 2 else float('nan')
    
    mcc = matthews_corrcoef(true_labels, predictions)
    
    prc_auc = average_precision_score(true_labels, logits[:, 1]) if len(np.unique(true_labels)) == 2 else float('nan')

    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_peft_model(model, lora_config):
    is_quantized = any(isinstance(module, nn.Linear) and "4bit" in str(type(module)) for module in model.modules())
    if is_quantized:
        print("is_q:true")
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_amp_fasta', type=str, required=True)
    parser.add_argument('--train_non_amp_fasta', type=str, required=True)
    parser.add_argument('--var_amp_fasta', type=str, required=True)
    parser.add_argument('--var_non_amp_fasta', type=str, required=True)
    parser.add_argument('--test_amp_fasta', type=str, required=True)
    parser.add_argument('--test_non_amp_fasta', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--model_shorthand', type=str, default="8m")
    parser.add_argument('--quantized', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)


    
    train_sequences, train_labels = prepare_dataset(args.train_amp_fasta, args.train_non_amp_fasta)
    var_sequences, var_labels = prepare_dataset(args.var_amp_fasta, args.var_non_amp_fasta)
    test_sequences, test_labels = prepare_dataset(args.test_amp_fasta, args.test_non_amp_fasta)

    
    model, tokenizer = load_esm2_model(args.model_shorthand, quantized=args.quantized)

    
    train_dataset = tokenize_and_create_dataset(train_sequences, train_labels, tokenizer)
    var_dataset = tokenize_and_create_dataset(var_sequences, var_labels, tokenizer)
    test_dataset = tokenize_and_create_dataset(test_sequences, test_labels, tokenizer)

    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="lora_only",
        target_modules=["query", "key", "value"]
    )
    model = apply_peft_model(model, lora_config)
    set_seed(42)

    if args.model_shorthand in ["8m", "35m"]:
        lr = 1e-3
    elif args.model_shorthand in ["150m"]:
        lr = 2e-4
    elif args.model_shorthand in ["650m", "3B"]:
        lr = 1e-4
    else:
        lr = 1e-4  
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        weight_decay=0.1,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=var_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("==== Start Training ====")
    trainer.train()
    pd.DataFrame(trainer.state.log_history).to_csv(f"{training_args.output_dir}/train_log.csv")

    
    os.makedirs(f"{args.output_dir}/model_complete", exist_ok=True)
    trainer.save_model(f"{args.output_dir}/model_complete")
    tokenizer.save_pretrained(f"{args.output_dir}/model_complete")


    print("==== Predict on Test Set ====")
    predictions = trainer.predict(test_dataset)
    
    logits = predictions.predictions
    true_labels = predictions.label_ids

    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    test_metrics = compute_metrics((logits, true_labels))

    
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(f"{args.output_dir}/test_metrics.csv", index=False)
    print(f"Saved all test metrics to {args.output_dir}/test_metrics.csv")

    


    
    np.save(f"{args.output_dir}/test_probabilities.npy", probabilities)
    np.save(f"{args.output_dir}/test_labels.npy", true_labels)
    print(f"Saved test metrics and predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
