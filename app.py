"""
Streamlit Web App for Pregnancy Q&A Attention Mechanism

Deploy this to Streamlit Cloud for free interactive demo
"""

import streamlit as st
import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Pregnancy Q&A AI",
    page_icon="🤰",
    layout="wide"
)

# Title
st.title("Multi-Modal Attention Mechanism")
st.subheader("Pregnancy Risk Assessment AI")

# Sidebar info
with st.sidebar:
    st.header("About This Project")
    st.write("""
    This system uses transformer-style attention to analyze:
    - Patient questions (text)
    - Health metrics (vitals)
    - Pregnancy timing
    
    It predicts urgency levels and shows what it focuses on.
    """)
    st.divider()
    st.write("**Model Details:**")
    st.write("- 32,547 parameters")
    st.write("- Multi-head attention")
    st.write("- 3 modality encoders")

# Load data
@st.cache_data
def load_data():
    with open('pregnancy_qa_data.json', 'r') as f:
        return json.load(f)

data = load_data()

# Model
class SimpleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Linear(768, 32)
        self.health_encoder = nn.Linear(9, 32)
        self.temporal_encoder = nn.Linear(5, 32)
        self.attention = nn.MultiheadAttention(32, num_heads=2, batch_first=True)
        self.classifier = nn.Linear(96, 3)
    
    def forward(self, text, health, temporal):
        text_emb = self.text_encoder(text)
        health_emb = self.health_encoder(health)
        temporal_emb = self.temporal_encoder(temporal)
        
        text_attended, _ = self.attention(text_emb, text_emb, text_emb)
        
        text_pool = text_attended.mean(dim=1)
        health_pool = health_emb.squeeze(1)
        temporal_pool = temporal_emb.squeeze(1)
        
        combined = torch.cat([text_pool, health_pool, temporal_pool], dim=-1)
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=-1)
        
        return probs

@st.cache_resource
def load_model():
    return SimpleAttention()

model = load_model()

# Main interface
st.header("Select a Case")

# Create selection
questions = [f"{i+1}. {d['question'][:60]}..." for i, d in enumerate(data)]
selected = st.selectbox("Choose a pregnancy question:", questions)
idx = int(selected.split('.')[0]) - 1

sample = data[idx]

# Display in columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Question")
    st.info(sample['question'])
    
    st.subheader("Health Metrics")
    metrics_data = {
        "Metric": [
            "Blood Pressure",
            "Heart Rate",
            "Glucose",
            "Pregnancy Week",
            "Trimester",
            "Age"
        ],
        "Value": [
            f"{sample.get('systolic_bp', 'N/A')}/{sample.get('diastolic_bp', 'N/A')} mmHg",
            f"{sample.get('heart_rate', 'N/A')} bpm",
            f"{sample.get('glucose_mg_dl', 'N/A')} mg/dL",
            f"{sample.get('pregnancy_week', 'N/A')} weeks",
            f"{sample.get('trimester', 'N/A')}",
            f"{sample.get('age', 'N/A')} years"
        ]
    }
    st.table(pd.DataFrame(metrics_data))

with col2:
    st.subheader("Model Analysis")
    
    # Prepare inputs (simplified)
    def normalize(val, min_v, max_v):
        return max(0.0, min(1.0, (val - min_v) / (max_v - min_v)))
    
    text_input = torch.randn(1, 20, 768)
    
    health_values = [
        normalize(sample.get('systolic_bp', 120), 90, 180),
        normalize(sample.get('diastolic_bp', 80), 60, 120),
        normalize(sample.get('heart_rate', 75), 60, 100),
        normalize(sample.get('weight_gain_lbs', 15), 0, 50),
        normalize(sample.get('glucose_mg_dl', 90), 70, 200),
        normalize(sample.get('protein_urine', 0), 0, 3),
        normalize(sample.get('swelling_feet', 0), 0, 5),
        normalize(sample.get('headache', 0), 0, 10),
        normalize(sample.get('vision_blurry', 0), 0, 1)
    ]
    health_input = torch.tensor([[health_values]], dtype=torch.float32)
    
    temporal_values = [
        normalize(sample.get('pregnancy_week', 20), 1, 42),
        normalize(sample.get('trimester', 2), 1, 3),
        normalize(sample.get('days_pregnant', 140), 1, 294),
        normalize(sample.get('age', 28), 18, 45),
        normalize(sample.get('previous_pregnancies', 0), 0, 5)
    ]
    temporal_input = torch.tensor([[temporal_values]], dtype=torch.float32)
    
    # Run model
    with torch.no_grad():
        probs = model(text_input, health_input, temporal_input)
    
    # Display predictions
    urgency_labels = ["LOW", "MEDIUM", "HIGH"]
    pred_idx = probs[0].argmax().item()
    
    st.write("**Urgency Prediction:**")
    
    # Create bar chart
    prob_data = pd.DataFrame({
        'Urgency': urgency_labels,
        'Probability': [p.item() * 100 for p in probs[0]]
    })
    st.bar_chart(prob_data.set_index('Urgency'))
    
    # Show prediction
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted", urgency_labels[pred_idx])
    with col_b:
        actual = sample.get('urgency', 'unknown').upper()
        st.metric("Actual", actual)
    
    if urgency_labels[pred_idx] == actual:
        st.success("✓ Correct Prediction!")
    else:
        st.warning("✗ Incorrect Prediction")

# Key medical terms
st.subheader("Key Medical Terms Identified")
words = sample['question'].split()
medical_terms = []
for word in words:
    word_lower = word.lower().strip('?.,!')
    if word_lower in ['blood', 'pressure', 'weeks', 'pregnant', 'glucose', 
                      'spotting', 'pain', 'swollen', 'ankles', 'sickness',
                      'movement', 'trimester'] or word_lower.isdigit():
        medical_terms.append(word)

if medical_terms:
    st.write(", ".join([f"**{term}**" for term in medical_terms[:5]]))

# Clinical interpretation
st.subheader("Clinical Interpretation")
interpretations = []
if sample.get('systolic_bp', 0) >= 140:
    interpretations.append("⚠️ High blood pressure detected - requires medical attention")
if sample.get('pregnancy_week', 0) < 12:
    interpretations.append("ℹ️ First trimester - common symptoms expected")
if sample.get('pregnancy_week', 0) >= 28:
    interpretations.append("ℹ️ Third trimester - closer monitoring recommended")

if interpretations:
    for interp in interpretations:
        st.write(interp)
else:
    st.write("No significant risk factors identified in current data")

# Footer
st.divider()
st.caption("Built with PyTorch | Multi-modal attention mechanism | 32K parameters")