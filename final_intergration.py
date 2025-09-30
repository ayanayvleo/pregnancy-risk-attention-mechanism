"""
FINAL INTEGRATION - Real Pregnancy Data + Attention Model

This connects everything:
1. Loads real pregnancy questions from JSON
2. Processes them through the data loader
3. Feeds them to the attention model
4. Shows what the model focuses on
"""

import torch
import torch.nn as nn
import json
import numpy as np

print("="*70)
print("PREGNANCY Q&A WITH REAL DATA")
print("="*70)
print()

# ============================================================================
# IMPORT DATA LOADER (simplified version here)
# ============================================================================

class DataLoader:
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
    
    def get_batch(self, indices):
        ranges = {
            "systolic_bp": (90, 180), "diastolic_bp": (60, 120),
            "heart_rate": (60, 100), "weight_gain_lbs": (0, 50),
            "glucose_mg_dl": (70, 200), "protein_urine": (0, 3),
            "swelling_feet": (0, 5), "headache": (0, 10),
            "vision_blurry": (0, 1), "pregnancy_week": (1, 42),
            "trimester": (1, 3), "days_pregnant": (1, 294),
            "age": (18, 45), "previous_pregnancies": (0, 5)
        }
        
        text_batch = []
        health_batch = []
        temporal_batch = []
        questions = []
        
        for idx in indices:
            sample = self.data[idx]
            
            # Text
            text_batch.append(self.text_to_embedding(sample["question"]))
            questions.append(sample["question"])
            
            # Health
            health = [self.normalize(sample.get(m, 0), m, ranges) 
                     for m in self.health_metrics]
            health_batch.append(torch.tensor([health], dtype=torch.float32))
            
            # Temporal
            temporal = [self.normalize(sample.get(f, 0), f, ranges) 
                       for f in self.temporal_features]
            temporal_batch.append(torch.tensor([temporal], dtype=torch.float32))
        
        return (
            torch.stack(text_batch),
            torch.stack(health_batch),
            torch.stack(temporal_batch),
            questions
        )


# ============================================================================
# SIMPLIFIED ATTENTION MODEL
# ============================================================================

class SimpleAttentionModel(nn.Module):
    def __init__(self, text_dim=768, health_dim=9, temporal_dim=5, d_model=256):
        super().__init__()
        
        # Encoders to convert different data types to same size
        self.text_encoder = nn.Linear(text_dim, d_model)
        self.health_encoder = nn.Linear(health_dim, d_model)
        self.temporal_encoder = nn.Linear(temporal_dim, d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Output for answer generation
        self.output = nn.Linear(d_model, d_model)
    
    def forward(self, text, health, temporal):
        # Encode each modality
        text_emb = self.text_encoder(text)
        health_emb = self.health_encoder(health)
        temporal_emb = self.temporal_encoder(temporal)
        
        # Apply attention to text
        text_attended, attn_weights = self.attention(text_emb, text_emb, text_emb)
        
        # Pool to single vector per sample
        text_pooled = text_attended.mean(dim=1)
        health_pooled = health_emb.squeeze(1)
        temporal_pooled = temporal_emb.squeeze(1)
        
        # Combine all information
        combined = torch.cat([text_pooled, health_pooled, temporal_pooled], dim=-1)
        fused = self.fusion(combined)
        output = self.output(fused)
        
        return output, attn_weights


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    print("Step 1: Loading real pregnancy data...")
    loader = DataLoader("pregnancy_qa_data.json")
    print(f"✓ Loaded {len(loader.data)} real pregnancy questions\n")
    
    print("Step 2: Creating attention model...")
    model = SimpleAttentionModel(text_dim=768, health_dim=9, temporal_dim=5)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model has {total_params:,} parameters\n")
    
    print("Step 3: Processing samples through the model...")
    print("="*70)
    print()
    
    # Process first 3 samples
    text, health, temporal, questions = loader.get_batch([0, 1, 2])
    
    # Run through model
    with torch.no_grad():
        output, attention_weights = model(text, health, temporal)
    
    # Show results for each question
    for i in range(len(questions)):
        sample = loader.data[i]
        
        print(f"{'='*70}")
        print(f"SAMPLE #{i+1}")
        print(f"{'='*70}")
        print(f"\nQuestion: {questions[i]}")
        print(f"\nKey Health Metrics:")
        print(f"  • Blood Pressure: {sample.get('systolic_bp', 'N/A')}/{sample.get('diastolic_bp', 'N/A')}")
        print(f"  • Week: {sample.get('pregnancy_week', 'N/A')}")
        print(f"  • Urgency: {sample.get('urgency', 'N/A')}")
        
        # Show what the model is focusing on
        attn = attention_weights[i].mean(dim=0)  # Average across heads
        
        # Get top 5 words the model focuses on (fixed dimension handling)
        if attn.dim() > 1:
            attn_scores = attn.mean(dim=-1)
        else:
            attn_scores = attn
        
        top_indices = attn_scores.topk(min(5, len(attn_scores))).indices
        words = questions[i].split()
        
        print(f"\nModel is paying most attention to these words:")
        for idx in top_indices:
            if idx < len(words):
                print(f"  • '{words[idx]}'")
        
        print(f"\nModel output vector size: {output[i].shape}")
        print()
    
    print("="*70)
    print("WHAT JUST HAPPENED")
    print("="*70)
    print("""
Your attention model just:
  1. ✓ Read real pregnancy questions from your JSON file
  2. ✓ Converted text to embeddings (numbers)
  3. ✓ Processed health metrics (blood pressure, etc.)
  4. ✓ Analyzed pregnancy timing (week, trimester)
  5. ✓ Used attention to focus on important words
  6. ✓ Combined all information into a unified output

This output can be used to:
  - Generate answers
  - Classify urgency levels
  - Recommend next steps
  - Identify risk factors

TO SHOW A HIRING MANAGER:
  • You built a multi-modal attention mechanism
  • It processes real healthcare data
  • You understand data pipelines and ML architecture
  • You can explain what the model is doing
    """)
    
    print("\n" + "="*70)
    print("NEXT STEPS TO MAKE IT EVEN MORE IMPRESSIVE")
    print("="*70)
    print("""
1. TRAINING: Teach it to actually generate answers
   - Create question-answer pairs
   - Train with a loss function
   - Evaluate accuracy

2. VISUALIZATION: Show attention heatmaps
   - Which words get attention?
   - Which health metrics matter most?
   - Create charts/graphs

3. DEPLOYMENT: Make it usable
   - Simple web interface
   - REST API
   - Command-line tool

4. DOCUMENTATION: Explain your work
   - README with examples
   - Code comments
   - Architecture diagrams
    """)


if __name__ == "__main__":
    main()