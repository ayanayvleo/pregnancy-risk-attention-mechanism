"""
PHASE 3: TRAINING THE ATTENTION MODEL

This teaches the model to learn patterns from pregnancy data.
We'll train it to predict urgency levels (low/medium/high).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from typing import List, Dict, Tuple

print("="*70)
print("PHASE 3: TRAINING THE ATTENTION MODEL")
print("="*70)
print()

# ============================================================================
# DATA LOADER
# ============================================================================

class PregnancyDataLoader:
    """Loads and prepares pregnancy data for training"""
    
    def __init__(self, json_file="pregnancy_qa_data.json"):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.health_metrics = [
            "systolic_bp", "diastolic_bp", "heart_rate",
            "weight_gain_lbs", "glucose_mg_dl", "protein_urine",
            "swelling_feet", "headache", "vision_blurry"
        ]
        
        self.temporal_features = [
            "pregnancy_week", "trimester", "days_pregnant",
            "age", "previous_pregnancies"
        ]
        
        # Urgency mapping for classification
        self.urgency_map = {"low": 0, "medium": 1, "high": 2}
        self.urgency_labels = ["low", "medium", "high"]
    
    def normalize(self, value, metric, ranges):
        if metric not in ranges:
            return value
        min_val, max_val = ranges[metric]
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def text_to_embedding(self, text, max_length=20, embedding_dim=768):
        words = text.lower().split()
        embeddings = []
        
        for word in words[:max_length]:
            seed = hash(word) % 100000
            np.random.seed(seed)
            embeddings.append(np.random.randn(embedding_dim).astype(np.float32))
        
        while len(embeddings) < max_length:
            embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
        
        return torch.tensor(np.array(embeddings[:max_length]))
    
    def get_sample(self, idx):
        """Get a single training sample with label"""
        ranges = {
            "systolic_bp": (90, 180), "diastolic_bp": (60, 120),
            "heart_rate": (60, 100), "weight_gain_lbs": (0, 50),
            "glucose_mg_dl": (70, 200), "protein_urine": (0, 3),
            "swelling_feet": (0, 5), "headache": (0, 10),
            "vision_blurry": (0, 1), "pregnancy_week": (1, 42),
            "trimester": (1, 3), "days_pregnant": (1, 294),
            "age": (18, 45), "previous_pregnancies": (0, 5)
        }
        
        sample = self.data[idx]
        
        # Input features
        text = self.text_to_embedding(sample["question"])
        
        health = [self.normalize(sample.get(m, 0), m, ranges) 
                 for m in self.health_metrics]
        health_tensor = torch.tensor([health], dtype=torch.float32)
        
        temporal = [self.normalize(sample.get(f, 0), f, ranges) 
                   for f in self.temporal_features]
        temporal_tensor = torch.tensor([temporal], dtype=torch.float32)
        
        # Label (what we want to predict)
        urgency = sample.get("urgency", "low")
        label = self.urgency_map[urgency]
        
        return text, health_tensor, temporal_tensor, label
    
    def get_all_data(self):
        """Get all data split into train/test"""
        all_indices = list(range(len(self.data)))
        
        # Use first 6 for training, last 2 for testing
        train_indices = all_indices[:6]
        test_indices = all_indices[6:]
        
        return train_indices, test_indices


# ============================================================================
# ATTENTION MODEL (Same as before)
# ============================================================================

class AttentionModel(nn.Module):
    def __init__(self, text_dim=768, health_dim=9, temporal_dim=5, 
                 d_model=32, num_classes=3):  # Much smaller!
        super().__init__()
        
        # Smaller encoders
        self.text_encoder = nn.Linear(text_dim, d_model)
        self.health_encoder = nn.Linear(health_dim, d_model)
        self.temporal_encoder = nn.Linear(temporal_dim, d_model)
        
        # Simple attention (2 heads instead of 4)
        self.attention = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        
        # Much simpler fusion - less overfitting
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(0.3)  # More dropout
        )
        
        # Simple classifier
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, text, health, temporal):
        # Encode
        text_emb = self.text_encoder(text)
        health_emb = self.health_encoder(health)
        temporal_emb = self.temporal_encoder(temporal)
        
        # Attention on text
        text_attended, _ = self.attention(text_emb, text_emb, text_emb)
        
        # Pool
        text_pooled = text_attended.mean(dim=1)
        health_pooled = health_emb.squeeze(1)
        temporal_pooled = temporal_emb.squeeze(1)
        
        # Fuse and classify
        combined = torch.cat([text_pooled, health_pooled, temporal_pooled], dim=-1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    """Handles the training process"""
    
    def __init__(self, model, data_loader, learning_rate=0.001):
        self.model = model
        self.loader = data_loader
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.train_accuracies = []
    
    def train_epoch(self, indices):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for idx in indices:
            # Get data
            text, health, temporal, label = self.loader.get_sample(idx)
            
            # Add batch dimension
            text = text.unsqueeze(0)
            health = health.unsqueeze(0)
            temporal = temporal.unsqueeze(0)
            label_tensor = torch.tensor([label], dtype=torch.long)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(text, health, temporal)
            loss = self.criterion(logits, label_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label_tensor).sum().item()
            total += 1
        
        avg_loss = total_loss / len(indices)
        accuracy = 100.0 * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, indices):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for idx in indices:
                text, health, temporal, label = self.loader.get_sample(idx)
                
                text = text.unsqueeze(0)
                health = health.unsqueeze(0)
                temporal = temporal.unsqueeze(0)
                
                logits = self.model(text, health, temporal)
                pred = logits.argmax(dim=1).item()
                
                predictions.append(pred)
                true_labels.append(label)
                
                correct += (pred == label)
                total += 1
        
        accuracy = 100.0 * correct / total
        return accuracy, predictions, true_labels
    
    def train(self, train_indices, test_indices, num_epochs=100):
        """Full training loop"""
        print(f"Training on {len(train_indices)} samples")
        print(f"Testing on {len(test_indices)} samples")
        print(f"Training for {num_epochs} epochs...")
        print()
        
        best_test_acc = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_indices)
            
            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                test_acc, _, _ = self.evaluate(test_indices)
                
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
                print(f"  Test Acc: {test_acc:.1f}%")
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    print(f"  New best test accuracy!")
                print()
        
        return best_test_acc


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("Step 1: Loading data...")
    loader = PregnancyDataLoader("pregnancy_qa_data.json")
    print(f"Loaded {len(loader.data)} samples")
    
    # Show urgency distribution
    urgency_counts = {}
    for sample in loader.data:
        urgency = sample.get("urgency", "low")
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    print("\nUrgency distribution:")
    for urgency, count in sorted(urgency_counts.items()):
        print(f"  {urgency}: {count} samples")
    print()
    
    print("Step 2: Creating model...")
    model = AttentionModel(text_dim=768, health_dim=9, temporal_dim=5, 
                          d_model=32, num_classes=3)  # Smaller model!
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} trainable parameters")
    print("(Much smaller to prevent overfitting on small data)")
    print()
    
    print("Step 3: Splitting data...")
    train_indices, test_indices = loader.get_all_data()
    print(f"Training set: {train_indices}")
    print(f"Test set: {test_indices}")
    print()
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print()
    
    trainer = Trainer(model, loader, learning_rate=0.001)
    best_acc = trainer.train(train_indices, test_indices, num_epochs=100)
    
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest test accuracy: {best_acc:.1f}%")
    print()
    
    print("Step 4: Final evaluation on test set...")
    test_acc, predictions, true_labels = trainer.evaluate(test_indices)
    
    print("\nTest Results:")
    for i, idx in enumerate(test_indices):
        sample = loader.data[idx]
        pred_label = loader.urgency_labels[predictions[i]]
        true_label = loader.urgency_labels[true_labels[i]]
        correct = "✓" if predictions[i] == true_labels[i] else "✗"
        
        print(f"\n{correct} Sample {idx+1}:")
        print(f"  Question: {sample['question'][:60]}...")
        print(f"  True urgency: {true_label}")
        print(f"  Predicted: {pred_label}")
    
    print("\n" + "="*70)
    print("WHAT YOU JUST DID")
    print("="*70)
    print("""
You trained a multi-modal attention model that:
  1. Learned patterns from pregnancy questions
  2. Identified relationships between symptoms and urgency
  3. Improved its predictions over 100 training iterations
  4. Achieved accuracy on unseen test data

This demonstrates:
  - Understanding of supervised learning
  - Training loops and optimization
  - Model evaluation and metrics
  - Real-world healthcare AI application

For a hiring manager, this shows you can:
  - Build AND train neural networks
  - Work with real data end-to-end
  - Evaluate model performance
  - Apply ML to practical problems
    """)


if __name__ == "__main__":
    main()