"""
DATA LOADER - Reads real pregnancy data and prepares it for the attention model

This is the bridge between your data files and your AI model!
"""

import torch
import json
import numpy as np
from typing import List, Dict, Tuple

print("="*70)
print("PREGNANCY DATA LOADER")
print("="*70)
print()

# ============================================================================
# STEP 1: LOAD THE JSON DATA
# ============================================================================

class PregnancyDataLoader:
    """
    Loads pregnancy Q&A data and converts it to model-ready tensors.
    
    This class handles:
    1. Reading JSON files
    2. Converting text to embeddings
    3. Normalizing health metrics
    4. Preparing batches for training
    """
    
    def __init__(self, json_file="pregnancy_qa_data.json"):
        print(f"Loading data from: {json_file}")
        
        # Load the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✓ Loaded {len(self.data)} samples")
        print()
        
        # Define which health metrics we'll use
        self.health_metrics = [
            "systolic_bp", "diastolic_bp", "heart_rate",
            "weight_gain_lbs", "glucose_mg_dl", "protein_urine",
            "swelling_feet", "headache", "vision_blurry"
        ]
        
        # Define temporal features
        self.temporal_features = [
            "pregnancy_week", "trimester", "days_pregnant",
            "age", "previous_pregnancies"
        ]
        
        # Normalization ranges (so all values are 0-1)
        self.norm_ranges = {
            "systolic_bp": (90, 180),
            "diastolic_bp": (60, 120),
            "heart_rate": (60, 100),
            "weight_gain_lbs": (0, 50),
            "glucose_mg_dl": (70, 200),
            "protein_urine": (0, 3),
            "swelling_feet": (0, 5),
            "headache": (0, 10),
            "vision_blurry": (0, 1),
            "pregnancy_week": (1, 42),
            "trimester": (1, 3),
            "days_pregnant": (1, 294),
            "age": (18, 45),
            "previous_pregnancies": (0, 5)
        }
    
    def normalize(self, value, metric_name):
        """
        Normalize a value to 0-1 range.
        
        Example: Blood pressure 120 with range (90, 180)
                 → (120-90)/(180-90) = 30/90 = 0.33
        """
        if metric_name not in self.norm_ranges:
            return value
        
        min_val, max_val = self.norm_ranges[metric_name]
        normalized = (value - min_val) / (max_val - min_val)
        
        # Clip to 0-1 range
        return max(0.0, min(1.0, normalized))
    
    def text_to_embedding(self, text: str, max_length=50, embedding_dim=768):
        """
        Convert text to embeddings.
        
        SIMPLE VERSION: Uses word-based random embeddings
        
        IN PRODUCTION, YOU'D USE:
        - from sentence_transformers import SentenceTransformer
        - model = SentenceTransformer('all-MiniLM-L6-v2')
        - embedding = model.encode(text)
        
        For now, we create consistent embeddings based on words.
        """
        words = text.lower().split()
        embeddings = []
        
        for i, word in enumerate(words[:max_length]):
            # Create consistent embedding for each word
            seed = hash(word) % 100000
            np.random.seed(seed)
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            embeddings.append(embedding)
        
        # Pad to max_length if needed
        while len(embeddings) < max_length:
            embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
        
        return torch.tensor(np.array(embeddings[:max_length]))
    
    def extract_health_metrics(self, sample: Dict) -> torch.Tensor:
        """
        Extract and normalize health metrics from a sample.
        
        Returns a tensor of shape (1, num_metrics)
        """
        values = []
        
        for metric in self.health_metrics:
            if metric in sample:
                value = sample[metric]
                normalized = self.normalize(value, metric)
                values.append(normalized)
            else:
                values.append(0.0)  # Missing value
        
        # Shape: (1, num_metrics) - the "1" is sequence length
        return torch.tensor([values], dtype=torch.float32)
    
    def extract_temporal_features(self, sample: Dict) -> torch.Tensor:
        """
        Extract and normalize temporal features from a sample.
        
        Returns a tensor of shape (1, num_features)
        """
        values = []
        
        for feature in self.temporal_features:
            if feature in sample:
                value = sample[feature]
                normalized = self.normalize(value, feature)
                values.append(normalized)
            else:
                values.append(0.0)
        
        return torch.tensor([values], dtype=torch.float32)
    
    def get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample ready for the model.
        
        Returns:
            text_embedding: (seq_len, 768)
            health_metrics: (1, num_metrics)
            temporal_features: (1, num_features)
            answer: The answer text (for reference)
        """
        sample = self.data[idx]
        
        text_emb = self.text_to_embedding(sample["question"])
        health = self.extract_health_metrics(sample)
        temporal = self.extract_temporal_features(sample)
        answer = sample["answer"]
        
        return text_emb, health, temporal, answer
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Get a batch of samples.
        
        Returns:
            text_batch: (batch_size, seq_len, 768)
            health_batch: (batch_size, 1, num_metrics)
            temporal_batch: (batch_size, 1, num_features)
            answers: List of answer texts
        """
        text_list = []
        health_list = []
        temporal_list = []
        answers = []
        
        for idx in indices:
            text, health, temporal, answer = self.get_sample(idx)
            text_list.append(text)
            health_list.append(health)
            temporal_list.append(temporal)
            answers.append(answer)
        
        return (
            torch.stack(text_list),
            torch.stack(health_list),
            torch.stack(temporal_list),
            answers
        )
    
    def display_sample(self, idx: int):
        """Display a sample in readable format"""
        sample = self.data[idx]
        
        print(f"{'='*70}")
        print(f"SAMPLE #{idx + 1}")
        print(f"{'='*70}")
        print(f"\n❓ QUESTION:")
        print(f"   {sample['question']}")
        print(f"\n💡 ANSWER:")
        print(f"   {sample['answer'][:150]}...")
        print(f"\n🏥 HEALTH METRICS (normalized):")
        
        for metric in self.health_metrics:
            if metric in sample:
                raw_value = sample[metric]
                normalized = self.normalize(raw_value, metric)
                print(f"   {metric}: {raw_value} → {normalized:.3f}")
        
        print(f"\n⏰ TEMPORAL INFO (normalized):")
        for feature in self.temporal_features:
            if feature in sample:
                raw_value = sample[feature]
                normalized = self.normalize(raw_value, feature)
                print(f"   {feature}: {raw_value} → {normalized:.3f}")
        
        print(f"\n🚨 Urgency: {sample['urgency']}")
        print()


# ============================================================================
# STEP 2: TEST THE DATA LOADER
# ============================================================================

def test_data_loader():
    """Test that everything works!"""
    
    print("="*70)
    print("TESTING DATA LOADER")
    print("="*70)
    print()
    
    # Create the loader
    loader = PregnancyDataLoader("pregnancy_qa_data.json")
    
    # Display first two samples
    print("Displaying sample data:\n")
    loader.display_sample(0)
    loader.display_sample(1)
    
    # Test getting a single sample
    print("="*70)
    print("TESTING SINGLE SAMPLE EXTRACTION")
    print("="*70)
    print()
    
    text, health, temporal, answer = loader.get_sample(0)
    
    print(f"✓ Text embedding shape: {text.shape}")
    print(f"  Expected: (50, 768) - 50 words, 768-dim embeddings")
    print()
    
    print(f"✓ Health metrics shape: {health.shape}")
    print(f"  Expected: (1, {len(loader.health_metrics)})")
    print(f"  Values: {health[0][:5].tolist()}")  # Show first 5 values
    print()
    
    print(f"✓ Temporal features shape: {temporal.shape}")
    print(f"  Expected: (1, {len(loader.temporal_features)})")
    print(f"  Values: {temporal[0].tolist()}")
    print()
    
    # Test getting a batch
    print("="*70)
    print("TESTING BATCH EXTRACTION")
    print("="*70)
    print()
    
    text_batch, health_batch, temporal_batch, answers = loader.get_batch([0, 1, 2])
    
    print(f"✓ Batch text shape: {text_batch.shape}")
    print(f"  (3 samples, 50 words each, 768-dim embeddings)")
    print()
    
    print(f"✓ Batch health shape: {health_batch.shape}")
    print(f"  (3 samples, 1 sequence length, {len(loader.health_metrics)} metrics)")
    print()
    
    print(f"✓ Batch temporal shape: {temporal_batch.shape}")
    print(f"  (3 samples, 1 sequence length, {len(loader.temporal_features)} features)")
    print()
    
    print(f"✓ Number of answers: {len(answers)}")
    print()
    
    # Show summary
    print("="*70)
    print("SUCCESS! DATA IS READY FOR THE MODEL!")
    print("="*70)
    print()
    print("What we can do now:")
    print("  ✓ Load 8 real pregnancy questions")
    print("  ✓ Convert text to embeddings")
    print("  ✓ Extract health metrics")
    print("  ✓ Normalize all values to 0-1 range")
    print("  ✓ Create batches for training")
    print()
    print("Next step: Feed this data to your attention model!")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    test_data_loader()