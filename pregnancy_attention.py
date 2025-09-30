import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("="*60)
print("PREGNANCY & MOTHERHOOD ATTENTION MECHANISM")
print("="*60)
print("\nImports successful! PyTorch version:", torch.__version__)
print()

# ============================================================================
# PART 1: BASIC ATTENTION MECHANISM
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    The fundamental attention mechanism.
    
    Think of it like: "Which parts of the information should I focus on?"
    - Query (Q): What am I looking for?
    - Key (K): What options are available?
    - Value (V): What is the actual content?
    """
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # Calculate attention scores: how relevant is each piece of info?
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Convert scores to probabilities (softmax)
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads looking at the data from different perspectives.
    
    Like having 8 different experts examining the same question:
    - Expert 1 focuses on blood pressure values
    - Expert 2 focuses on pregnancy week
    - Expert 3 focuses on symptom patterns
    - etc.
    """
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear projections for queries, keys, values
        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Split into multiple heads
        q = self.w_qs(q).view(batch_size, len_q, n_heads, d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, n_heads, d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_v, n_heads, d_v).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # Apply attention
        output, attn = self.attention(q, k, v, mask=mask)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual
        output = self.layer_norm(output)
        
        return output, attn


# ============================================================================
# PART 2: MULTI-MODAL COMPONENTS
# ============================================================================

class ModalityEncoder(nn.Module):
    """
    Converts different types of data into a common format.
    
    Example:
    - Text: "Is my blood pressure normal?" → [embedding vector]
    - Health data: [120, 80, 98.6, ...] → [embedding vector]
    - Time data: [week 20, trimester 2] → [embedding vector]
    
    All become the same size so they can "talk" to each other!
    """
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.layer_norm(self.encoder(x))


class CrossModalAttention(nn.Module):
    """
    Lets one type of data pay attention to another type.
    
    Example: The question "Is my blood pressure normal at week 20?"
    can look at both the blood pressure readings AND the pregnancy week
    to get context.
    """
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
    
    def forward(self, query_modality, key_value_modality, mask=None):
        output, attn = self.attention(query_modality, key_value_modality, 
                                      key_value_modality, mask)
        return output, attn


# ============================================================================
# PART 3: COMPLETE MULTI-MODAL SYSTEM
# ============================================================================

class MultiModalAttentionQA(nn.Module):
    """
    The complete system that combines:
    1. Text (questions)
    2. Health data (vitals, symptoms)
    3. Temporal info (pregnancy week, trimester)
    """
    def __init__(self, 
                 text_dim=768,
                 health_dim=20,
                 temporal_dim=10,
                 d_model=512,
                 n_heads=8,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        d_k = d_v = d_model // n_heads
        
        print(f"Initializing Multi-Modal Attention System:")
        print(f"  - Text dimension: {text_dim}")
        print(f"  - Health metrics: {health_dim}")
        print(f"  - Temporal features: {temporal_dim}")
        print(f"  - Model dimension: {d_model}")
        print(f"  - Attention heads: {n_heads}")
        print()
        
        # Encoders for each modality
        self.text_encoder = ModalityEncoder(text_dim, d_model, dropout)
        self.health_encoder = ModalityEncoder(health_dim, d_model, dropout)
        self.temporal_encoder = ModalityEncoder(temporal_dim, d_model, dropout)
        
        # Self-attention within modalities
        self.text_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.health_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        
        # Cross-modal attention
        self.text_to_health_attn = CrossModalAttention(n_heads, d_model, d_k, d_v, dropout)
        self.health_to_text_attn = CrossModalAttention(n_heads, d_model, d_k, d_v, dropout)
        self.temporal_cross_attn = CrossModalAttention(n_heads, d_model, d_k, d_v, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, text_input, health_input, temporal_input, 
                text_mask=None, health_mask=None):
        batch_size = text_input.size(0)
        
        # Step 1: Encode each modality
        text_emb = self.text_encoder(text_input)
        health_emb = self.health_encoder(health_input)
        temporal_emb = self.temporal_encoder(temporal_input)
        
        # Step 2: Self-attention within modalities
        text_self, text_self_attn = self.text_self_attn(text_emb, text_emb, text_emb, text_mask)
        health_self, health_self_attn = self.health_self_attn(health_emb, health_emb, health_emb, health_mask)
        
        # Step 3: Cross-modal attention
        text_to_health, t2h_attn = self.text_to_health_attn(text_self, health_self, health_mask)
        health_to_text, h2t_attn = self.health_to_text_attn(health_self, text_self, text_mask)
        
        # Step 4: Temporal context
        temporal_context, temp_attn = self.temporal_cross_attn(
            torch.cat([text_to_health, health_to_text], dim=1),
            temporal_emb
        )
        
        # Step 5: Pool and fuse
        text_pooled = text_to_health.mean(dim=1)
        health_pooled = health_to_text.mean(dim=1)
        temporal_pooled = temporal_context.mean(dim=1)
        
        fused = torch.cat([text_pooled, health_pooled, temporal_pooled], dim=-1)
        fused_output = self.fusion(fused)
        fused_output = self.output_proj(fused_output)
        
        # Return output and attention weights for interpretability
        attention_weights = {
            'text_self_attention': text_self_attn,
            'health_self_attention': health_self_attn,
            'text_to_health_attention': t2h_attn,
            'health_to_text_attention': h2t_attn,
            'temporal_attention': temp_attn
        }
        
        return fused_output, attention_weights


class PregnancyQASystem(nn.Module):
    """
    Complete question-answering system wrapper.
    """
    def __init__(self, vocab_size=30000, max_answer_len=100, **attention_kwargs):
        super().__init__()
        
        self.attention = MultiModalAttentionQA(**attention_kwargs)
        self.answer_decoder = nn.LSTM(
            input_size=attention_kwargs.get('d_model', 512),
            hidden_size=attention_kwargs.get('d_model', 512),
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        self.output_projection = nn.Linear(attention_kwargs.get('d_model', 512), vocab_size)
        self.max_answer_len = max_answer_len
    
    def forward(self, text_input, health_input, temporal_input, 
                text_mask=None, health_mask=None):
        # Get multi-modal context
        context, attn_weights = self.attention(
            text_input, health_input, temporal_input,
            text_mask, health_mask
        )
        
        # Decode answer
        context_expanded = context.unsqueeze(1).repeat(1, self.max_answer_len, 1)
        decoder_out, _ = self.answer_decoder(context_expanded)
        logits = self.output_projection(decoder_out)
        
        return logits, attn_weights


# ============================================================================
# PART 4: DEMONSTRATION
# ============================================================================

def run_demo():
    print("="*60)
    print("RUNNING DEMONSTRATION")
    print("="*60)
    print()
    
    # Configuration
    batch_size = 4
    seq_len_text = 20
    seq_len_health = 10
    seq_len_temporal = 5
    
    print("Creating model...")
    model = PregnancyQASystem(
        vocab_size=30000,
        text_dim=768,
        health_dim=20,
        temporal_dim=10,
        d_model=512,
        n_heads=8
    )
    
    print("\nGenerating example data...")
    print(f"  Batch size: {batch_size} (processing 4 questions simultaneously)")
    print(f"  Text sequence length: {seq_len_text} (20 words per question)")
    print(f"  Health data points: {seq_len_health} (10 health measurements)")
    print(f"  Temporal features: {seq_len_temporal} (5 time-related features)")
    print()
    
    # Create example inputs
    text_input = torch.randn(batch_size, seq_len_text, 768)
    health_input = torch.randn(batch_size, seq_len_health, 20)
    temporal_input = torch.randn(batch_size, seq_len_temporal, 10)
    
    print("Running forward pass through the model...")
    output_logits, attention_weights = model(text_input, health_input, temporal_input)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print()
    print(f"✓ Output shape: {output_logits.shape}")
    print(f"  This means: {batch_size} answers, each up to 100 words long,")
    print(f"  choosing from a vocabulary of 30,000 words")
    print()
    
    print("✓ Attention mechanisms computed:")
    for name, attn in attention_weights.items():
        print(f"  - {name}: {attn.shape}")
    print()
    
    print("="*60)
    print("WHAT THE ATTENTION WEIGHTS TELL US")
    print("="*60)
    print()
    print("The attention weights show which information the model")
    print("focuses on when answering questions:")
    print()
    print("  • text_to_health_attention:")
    print("    Shows which health metrics are relevant to the question")
    print()
    print("  • health_to_text_attention:")
    print("    Shows which parts of the question relate to the health data")
    print()
    print("  • temporal_attention:")
    print("    Shows how pregnancy timing influences the answer")
    print()
    
    print("="*60)
    print("SUCCESS! The attention mechanism is working!")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Add real pregnancy data")
    print("  2. Train the model on question-answer pairs")
    print("  3. Visualize what the model pays attention to")
    print("  4. Deploy for real pregnancy Q&A")
    print()


if __name__ == "__main__":
    run_demo()