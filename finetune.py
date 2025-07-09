#!/usr/bin/env python3
"""
Fine-tuning script for binary sentiment analysis
Usage: python finetune.py --data data.jsonl --epochs 3 --lr 3e-5
"""

import argparse
import json
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """Load data from JSONL file"""
    texts = []
    labels = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
            # Convert string labels to integers
            label = 1 if data['label'].lower() == 'positive' else 0
            labels.append(label)
    
    return texts, labels

def train_epoch(model, dataloader, optimizer, scheduler, device, clip_value=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

def main():
    parser = argparse.ArgumentParser(description='Fine-tune sentiment analysis model')
    parser.add_argument('--data', required=True, help='Path to training data (JSONL format)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--model_name', default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                       help='Pre-trained model name')
    parser.add_argument('--output_dir', default='./model', help='Output directory for fine-tuned model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading data from {args.data}")
    texts, labels = load_data(args.data)
    logger.info(f"Loaded {len(texts)} samples")

    # Split data
    split_idx = int(len(texts) * (1 - args.validation_split))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    logger.info(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2  # Binary classification
    )
    model.to(device)

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    best_val_accuracy = 0

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, device)

        # Validate
        val_loss, val_acc, val_preds, val_true = evaluate(model, val_dataloader, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            logger.info(f"New best validation accuracy: {val_acc:.4f}")

            # Save model and tokenizer
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"Model saved to {args.output_dir}")

    # Final evaluation report
    logger.info("\nFinal Classification Report:")
    print(classification_report(val_true, val_preds, target_names=['negative', 'positive']))

if __name__ == "__main__":
    main()
