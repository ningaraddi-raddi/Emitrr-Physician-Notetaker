
# src/multitask_finetune.py - REGULARIZED VERSION
# -------------------------------------------------
#  Prevents overfitting with multiple techniques
# -------------------------------------------------
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from datasets import Dataset, DatasetDict

# ------------------- CONFIG -------------------
DATA_PATH   = "data/sentiment_intent_labeled_data.csv"
MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.1"
OUTPUT_DIR  = "outputs/multitask_regularized"
MAX_LENGTH  = 128
BATCH_SIZE  = 8           # Smaller batch size (was 16)
EPOCHS      = 3           # Fewer epochs (was 4)
LR          = 5e-6        # Much lower learning rate (was 2e-5)
WEIGHT_DECAY = 0.1        # Higher weight decay (was 0.01)
DROPOUT     = 0.3         # Higher dropout (was 0.1)
WARMUP_RATIO = 0.1        # Add warmup
SEED        = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- 1. LOAD & SPLIT -------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# label → id
sentiment2id = {l: i for i, l in enumerate(sorted(df["sentiment"].unique()))}
intent2id    = {l: i for i, l in enumerate(sorted(df["intent"].unique()))}
id2sentiment = {v: k for k, v in sentiment2id.items()}
id2intent    = {v: k for k, v in intent2id.items()}

df["sentiment_label"] = df["sentiment"].map(sentiment2id)
df["intent_label"]    = df["intent"].map(intent2id)

print(f"Sentiment classes: {sentiment2id}")
print(f"Intent classes:    {intent2id}")

# 70% train / 15% val / 15% test
train_val_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["sentiment_label"], random_state=SEED
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,
    stratify=train_val_df["sentiment_label"],
    random_state=SEED,
)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# ------------------- 2. DATA AUGMENTATION -------------------
def augment_text(text, p=0.1):
    """Simple word dropout augmentation"""
    words = text.split()
    if len(words) <= 3:
        return text
    
    # Randomly drop some words
    mask = np.random.random(len(words)) > p
    augmented = ' '.join([w for w, m in zip(words, mask) if m])
    
    return augmented if augmented.strip() else text

# Apply augmentation to training data only
print("\nApplying data augmentation...")
train_df_aug = train_df.copy()
train_df_aug["text"] = train_df_aug["text"].apply(lambda x: augment_text(x, p=0.15))

# Combine original and augmented
train_df_combined = pd.concat([train_df, train_df_aug]).reset_index(drop=True)
print(f"Training samples after augmentation: {len(train_df_combined)}")

# ------------------- 3. TOKENIZER -------------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df_combined.reset_index(drop=True)),
    "val":   Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
})

tokenized = dataset.map(tokenize_function, batched=True)

columns_to_keep = ["input_ids", "attention_mask", "sentiment_label", "intent_label"]
tokenized = tokenized.remove_columns(
    [c for c in tokenized["train"].column_names if c not in columns_to_keep]
)

tokenized = tokenized.rename_column("sentiment_label", "labels_sentiment")
tokenized = tokenized.rename_column("intent_label", "labels_intent")

tokenized.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels_sentiment", "labels_intent"],
)

# ------------------- 4. IMPROVED MODEL WITH REGULARIZATION -------------------
class MultiTaskBioBERT(nn.Module):
    def __init__(self, model_name, n_sent, n_int, dropout=0.3, freeze_layers=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        
        # Freeze early BERT layers to prevent overfitting
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                if hasattr(self.bert.encoder, 'layer'):
                    for param in self.bert.encoder.layer[i].parameters():
                        param.requires_grad = False
        
        # Higher dropout
        self.dropout = nn.Dropout(dropout)
        
        # Add intermediate layers with batch norm for better regularization
        self.sent_intermediate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sent_head = nn.Linear(hidden // 2, n_sent)
        
        self.int_intermediate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.int_head = nn.Linear(hidden // 2, n_int)

    def forward(self, input_ids, attention_mask,
                labels_sentiment=None, labels_intent=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        
        # Pass through intermediate layers
        sent_features = self.sent_intermediate(pooled)
        logits_s = self.sent_head(sent_features)
        
        int_features = self.int_intermediate(pooled)
        logits_i = self.int_head(int_features)

        loss = None
        if labels_sentiment is not None and labels_intent is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Weighted combination - can adjust weights
            loss_s = loss_fct(logits_s, labels_sentiment)
            loss_i = loss_fct(logits_i, labels_intent)
            loss = 0.5 * loss_s + 0.5 * loss_i  # Equal weight

        return ModelOutput(
            loss=loss,
            logits_sentiment=logits_s,
            logits_intent=logits_i,
        )

model = MultiTaskBioBERT(
    MODEL_NAME,
    n_sent=len(sentiment2id),
    n_int=len(intent2id),
    dropout=DROPOUT,
    freeze_layers=6  # Freeze first 6 layers
)

print(f"\n[INFO] Model parameters:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

# ------------------- 5. METRICS -------------------
def compute_metrics(eval_pred):
    logits_s, logits_i = eval_pred.predictions
    labels_s, labels_i = eval_pred.label_ids

    pred_s = np.argmax(logits_s, axis=1)
    pred_i = np.argmax(logits_i, axis=1)

    return {
        "sentiment_acc": accuracy_score(labels_s, pred_s),
        "intent_acc":    accuracy_score(labels_i, pred_i),
        "sentiment_f1":  f1_score(labels_s, pred_s, average="weighted"),
        "intent_f1":     f1_score(labels_i, pred_i, average="weighted"),
    }

# ------------------- 6. CUSTOM TRAINER -------------------
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels_s = inputs.pop("labels_sentiment")
        labels_i = inputs.pop("labels_intent")
        
        outputs = model(**inputs, labels_sentiment=labels_s, labels_intent=labels_i)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        
        all_preds_s = []
        all_preds_i = []
        all_labels_s = []
        all_labels_i = []
        total_loss = 0.0
        num_batches = 0
        
        for step, inputs in enumerate(eval_dataloader):
            labels_s = inputs.pop("labels_sentiment").to(self.args.device)
            labels_i = inputs.pop("labels_intent").to(self.args.device)
            
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = model(**inputs, labels_sentiment=labels_s, labels_intent=labels_i)
                loss = outputs.loss
                logits_s = outputs.logits_sentiment
                logits_i = outputs.logits_intent
            
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            
            all_preds_s.append(logits_s.cpu().numpy())
            all_preds_i.append(logits_i.cpu().numpy())
            all_labels_s.append(labels_s.cpu().numpy())
            all_labels_i.append(labels_i.cpu().numpy())
        
        all_preds_s = np.concatenate(all_preds_s, axis=0)
        all_preds_i = np.concatenate(all_preds_i, axis=0)
        all_labels_s = np.concatenate(all_labels_s, axis=0)
        all_labels_i = np.concatenate(all_labels_i, axis=0)
        
        from dataclasses import dataclass
        
        @dataclass
        class EvalPrediction:
            predictions: tuple
            label_ids: tuple
        
        eval_pred = EvalPrediction(
            predictions=(all_preds_s, all_preds_i),
            label_ids=(all_labels_s, all_labels_i)
        )
        
        metrics = self.compute_metrics(eval_pred)
        
        if num_batches > 0:
            metrics[f"{metric_key_prefix}_loss"] = total_loss / num_batches
        
        metrics = {
            f"{metric_key_prefix}_{k}" if not k.startswith(f"{metric_key_prefix}_") else k: v 
            for k, v in metrics.items()
        }
        
        self.log(metrics)
        
        return metrics

# ------------------- 7. TRAINING ARGUMENTS WITH REGULARIZATION -------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,  # Add warmup
    load_best_model_at_end=True,
    metric_for_best_model="eval_sentiment_acc",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=2,  # Keep only 2 best checkpoints
    report_to="none",
    seed=SEED,
    dataloader_pin_memory=False,
    # Additional regularization
    lr_scheduler_type="cosine",  # Cosine learning rate decay
    gradient_accumulation_steps=2,  # Simulate larger batch size
    max_grad_norm=1.0,  # Gradient clipping
    dataloader_drop_last=True,  # Drop incomplete batches
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    compute_metrics=compute_metrics,
)

# ------------------- 8. TRAIN -------------------
print("\n" + "="*60)
print("Starting training with regularization...")
print("="*60)
print(f"Learning Rate: {LR}")
print(f"Dropout: {DROPOUT}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Frozen Layers: 6/12")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("Training finished!")
print("="*60)

# ------------------- 9. EVALUATION -------------------
print("\n=== Validation Set Evaluation ===")
val_results = trainer.evaluate(tokenized["val"], metric_key_prefix="eval")
print(f"Sentiment Accuracy: {val_results['eval_sentiment_acc']:.4f}")
print(f"Intent Accuracy:    {val_results['eval_intent_acc']:.4f}")
print(f"Sentiment F1:       {val_results['eval_sentiment_f1']:.4f}")
print(f"Intent F1:          {val_results['eval_intent_f1']:.4f}")

print("\n=== Test Set Evaluation ===")
test_results = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
print(f"Sentiment Accuracy: {test_results['test_sentiment_acc']:.4f}")
print(f"Intent Accuracy:    {test_results['test_intent_acc']:.4f}")
print(f"Sentiment F1:       {test_results['test_sentiment_f1']:.4f}")
print(f"Intent F1:          {test_results['test_intent_f1']:.4f}")

# Check for overfitting
val_acc = (val_results['eval_sentiment_acc'] + val_results['eval_intent_acc']) / 2
test_acc = (test_results['test_sentiment_acc'] + test_results['test_intent_acc']) / 2
overfit_gap = val_acc - test_acc

print(f"\n Overfitting Analysis:")
print(f"  Val Avg Accuracy:  {val_acc:.4f}")
print(f"  Test Avg Accuracy: {test_acc:.4f}")
print(f"  Gap: {overfit_gap:.4f}")
if abs(overfit_gap) < 0.05:
    print("  Good generalization!")
elif abs(overfit_gap) < 0.10:
    print("  Slight overfitting")
else:
    print("  Significant overfitting")

# ------------------- 10. DETAILED REPORT -------------------
print("\n" + "="*60)
print("Generating detailed test report...")
print("="*60)

model.eval()
all_preds_s = []
all_preds_i = []
all_labels_s = []
all_labels_i = []

test_dataloader = trainer.get_eval_dataloader(tokenized["test"])
for batch in test_dataloader:
    labels_s = batch.pop("labels_sentiment").to(trainer.args.device)
    labels_i = batch.pop("labels_intent").to(trainer.args.device)
    
    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
    
    all_preds_s.append(outputs.logits_sentiment.cpu().numpy())
    all_preds_i.append(outputs.logits_intent.cpu().numpy())
    all_labels_s.append(labels_s.cpu().numpy())
    all_labels_i.append(labels_i.cpu().numpy())

logits_s = np.concatenate(all_preds_s, axis=0)
logits_i = np.concatenate(all_preds_i, axis=0)
true_s = np.concatenate(all_labels_s, axis=0)
true_i = np.concatenate(all_labels_i, axis=0)

pred_s = np.argmax(logits_s, axis=1)
pred_i = np.argmax(logits_i, axis=1)

print("\n--- Sentiment Classification Report ---")
print(classification_report(true_s, pred_s, target_names=list(sentiment2id.keys())))

print("\n--- Intent Classification Report ---")
print(classification_report(true_i, pred_i, target_names=list(intent2id.keys())))

print("\n--- Confusion Matrix (Sentiment) ---")
cm_sent = confusion_matrix(true_s, pred_s)
print(cm_sent)
print(f"Classes: {list(sentiment2id.keys())}")

print("\n--- Confusion Matrix (Intent) ---")
cm_int = confusion_matrix(true_i, pred_i)
print(cm_int)
print(f"Classes: {list(intent2id.keys())}")

# ------------------- 11. SAVE MODEL -------------------
final_path = f"{OUTPUT_DIR}/final_model"
os.makedirs(final_path, exist_ok=True)
model.bert.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

import json
with open(f"{final_path}/label_mappings.json", "w") as f:
    json.dump({
        "sentiment2id": sentiment2id,
        "intent2id": intent2id,
        "id2sentiment": id2sentiment,
        "id2intent": id2intent
    }, f, indent=2)

# Save full model for inference
torch.save(model.state_dict(), f"{final_path}/full_model.pt")

print(f"\n Model saved to {final_path}")

# ------------------- 12. INFERENCE -------------------
def predict(text):
    """Predict sentiment and intent for given text"""
    enc = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True,
        padding=True, 
        max_length=MAX_LENGTH,
        return_token_type_ids=False
    )
    
    enc = {k: v.to(model.bert.device) for k, v in enc.items()}
    
    model.eval()
    with torch.no_grad():
        out = model(**enc)
    
    sentiment_idx = out.logits_sentiment.argmax().item()
    intent_idx = out.logits_intent.argmax().item()
    
    sentiment_probs = torch.softmax(out.logits_sentiment, dim=1)[0]
    intent_probs = torch.softmax(out.logits_intent, dim=1)[0]
    
    return {
        "Sentiment": id2sentiment[sentiment_idx],
        "Sentiment_Confidence": f"{sentiment_probs[sentiment_idx].item():.2%}",
        "Intent": id2intent[intent_idx],
        "Intent_Confidence": f"{intent_probs[intent_idx].item():.2%}",
    }

# ------------------- 13. DEMO -------------------
print("\n" + "="*60)
print("Demo Predictions")
print("="*60)

demo_texts = [
    "I'm worried about my back pain but hope it improves soon.",
    "The treatment is working well and I feel much better.",
    "I have neck pain and stiffness after the accident.",
    "Thank you doctor, I'm relieved to hear that.",
]

for text in demo_texts:
    result = predict(text)
    print(f"\nText: '{text}'")
    print(f"  → Sentiment: {result['Sentiment']} ({result['Sentiment_Confidence']})")
    print(f"  → Intent: {result['Intent']} ({result['Intent_Confidence']})")

print("\n" + "="*60)
print("Training Complete with Regularization! ")
print("="*60)











#using pre-trained models without finetuning


# import json
# from transformers import pipeline

# # Initialize pretrained zero-shot model
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Define classification labels
# sentiment_labels = ["Anxious", "Neutral", "Reassured"]
# intent_labels = [
#     "Reporting symptoms",
#     "Seeking reassurance",
#     "Expressing concern",
#     "Confirming improvement"
# ]

# def analyze_sentiment_intent(text):
#     """Predict sentiment and intent using pretrained zero-shot model."""
#     if not isinstance(text, str) or len(text.strip()) == 0:
#         return {"Sentiment": "Unknown", "Intent": "Unknown"}

#     # Sentiment prediction
#     sentiment_result = classifier(text, sentiment_labels)
#     sentiment = sentiment_result["labels"][0]

#     # Intent prediction
#     intent_result = classifier(text, intent_labels)
#     intent = intent_result["labels"][0]

#     return {"Sentiment": sentiment, "Intent": intent}


# def run_pipeline(
#     input_path="data/processed/patient_text.txt",
#     output_path="data/processed/sentiment_results.json"
# ):
#     """Run sentiment & intent prediction for patient transcript."""
#     with open(input_path, "r", encoding="utf-8") as f:
#         text = f.read().strip()

#     print(" Processing text for sentiment and intent detection...\n")
#     result = analyze_sentiment_intent(text)

#     # Save output
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=4)

#     print(f" Predictions saved to: {output_path}")
#     print("\n Output:\n", json.dumps(result, indent=4))


# if __name__ == "__main__":
#     run_pipeline()
