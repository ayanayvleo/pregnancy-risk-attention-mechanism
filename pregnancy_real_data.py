"""
REAL PREGNANCY DATA CREATOR
Creates a dataset you can actually use and expand
"""

import json
import csv

print("="*70)
print("CREATING REAL PREGNANCY Q&A DATASET")
print("="*70)
print()

# ============================================================================
# REAL PREGNANCY DATA
# ============================================================================

pregnancy_data = [
    {
        "id": 1,
        "question": "Is my blood pressure of 140/90 dangerous at 28 weeks pregnant?",
        "answer": "Yes, this requires immediate medical attention. Blood pressure of 140/90 or higher is considered hypertension in pregnancy and can indicate preeclampsia, especially in the third trimester. Contact your healthcare provider right away.",
        
        # Health metrics (all the measurements)
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "heart_rate": 88,
        "weight_lbs": 165,
        "weight_gain_lbs": 22,
        "glucose_mg_dl": 98,
        "protein_urine": 2,  # 0=none, 1=trace, 2=1+, 3=2+
        "swelling_feet": 3,  # 0-5 scale
        "headache": 7,  # 0-10 scale
        "vision_blurry": 1,  # 0=no, 1=yes
        
        # Pregnancy timing
        "pregnancy_week": 28,
        "trimester": 3,
        "days_pregnant": 196,
        "due_date_days_away": 84,
        
        # Patient info
        "age": 34,
        "previous_pregnancies": 1,
        "bmi": 28.5,
        
        # Risk category (for the model to learn)
        "urgency": "high"  # low, medium, high
    },
    
    {
        "id": 2,
        "question": "I'm 12 weeks pregnant and having morning sickness. Is this normal?",
        "answer": "Yes, morning sickness at 12 weeks is completely normal. About 70% of pregnant women experience nausea and vomiting in the first trimester. It usually peaks around 9 weeks and improves by 14-16 weeks. Stay hydrated and eat small, frequent meals.",
        
        "systolic_bp": 118,
        "diastolic_bp": 75,
        "heart_rate": 76,
        "weight_lbs": 142,
        "weight_gain_lbs": 3,
        "glucose_mg_dl": 88,
        "protein_urine": 0,
        "swelling_feet": 0,
        "headache": 2,
        "vision_blurry": 0,
        "nausea_episodes_per_day": 6,
        "vomiting_episodes_per_day": 2,
        
        "pregnancy_week": 12,
        "trimester": 1,
        "days_pregnant": 84,
        "due_date_days_away": 196,
        
        "age": 28,
        "previous_pregnancies": 0,
        "bmi": 23.2,
        
        "urgency": "low"
    },
    
    {
        "id": 3,
        "question": "Should I be worried about swollen ankles at 36 weeks?",
        "answer": "Mild swelling in ankles and feet is common in late pregnancy due to increased fluid and pressure. However, if accompanied by high blood pressure, severe headaches, or sudden weight gain, contact your provider as these can indicate preeclampsia.",
        
        "systolic_bp": 125,
        "diastolic_bp": 80,
        "heart_rate": 82,
        "weight_lbs": 178,
        "weight_gain_lbs": 32,
        "glucose_mg_dl": 94,
        "protein_urine": 0,
        "swelling_feet": 4,
        "headache": 1,
        "vision_blurry": 0,
        
        "pregnancy_week": 36,
        "trimester": 3,
        "days_pregnant": 252,
        "due_date_days_away": 28,
        
        "age": 30,
        "previous_pregnancies": 1,
        "bmi": 26.8,
        
        "urgency": "low"
    },
    
    {
        "id": 4,
        "question": "I haven't felt my baby move much today at 30 weeks. What should I do?",
        "answer": "Decreased fetal movement requires immediate evaluation. Do a kick count: drink cold water, lie on your left side, and count movements for 2 hours. If fewer than 10 movements, call your provider or go to labor and delivery immediately.",
        
        "systolic_bp": 120,
        "diastolic_bp": 78,
        "heart_rate": 80,
        "weight_lbs": 168,
        "weight_gain_lbs": 26,
        "glucose_mg_dl": 90,
        "protein_urine": 0,
        "swelling_feet": 2,
        "headache": 0,
        "vision_blurry": 0,
        "fetal_movements_last_2hrs": 4,  # Should be 10+
        
        "pregnancy_week": 30,
        "trimester": 3,
        "days_pregnant": 210,
        "due_date_days_away": 70,
        
        "age": 26,
        "previous_pregnancies": 0,
        "bmi": 24.5,
        
        "urgency": "high"
    },
    
    {
        "id": 5,
        "question": "Can I exercise at 20 weeks pregnant?",
        "answer": "Yes! Exercise is generally safe and beneficial during pregnancy. Good options include walking, swimming, prenatal yoga, and light strength training. Avoid contact sports, activities with fall risk, and overheating. Listen to your body and stay hydrated.",
        
        "systolic_bp": 115,
        "diastolic_bp": 72,
        "heart_rate": 74,
        "weight_lbs": 152,
        "weight_gain_lbs": 12,
        "glucose_mg_dl": 86,
        "protein_urine": 0,
        "swelling_feet": 0,
        "headache": 0,
        "vision_blurry": 0,
        "energy_level": 7,  # 0-10 scale
        
        "pregnancy_week": 20,
        "trimester": 2,
        "days_pregnant": 140,
        "due_date_days_away": 140,
        
        "age": 29,
        "previous_pregnancies": 0,
        "bmi": 22.8,
        
        "urgency": "low"
    },
    
    {
        "id": 6,
        "question": "My glucose test came back at 145. Do I have gestational diabetes?",
        "answer": "A glucose reading of 145 mg/dL on the 1-hour screening test is borderline (normal is under 140). You'll likely need a 3-hour glucose tolerance test to confirm. If diagnosed with gestational diabetes, it's manageable with diet, exercise, and sometimes medication.",
        
        "systolic_bp": 122,
        "diastolic_bp": 76,
        "heart_rate": 78,
        "weight_lbs": 172,
        "weight_gain_lbs": 20,
        "glucose_mg_dl": 145,
        "protein_urine": 0,
        "swelling_feet": 1,
        "headache": 0,
        "vision_blurry": 0,
        
        "pregnancy_week": 26,
        "trimester": 2,
        "days_pregnant": 182,
        "due_date_days_away": 98,
        
        "age": 35,
        "previous_pregnancies": 2,
        "bmi": 29.2,
        
        "urgency": "medium"
    },
    
    {
        "id": 7,
        "question": "Is spotting normal at 8 weeks pregnant?",
        "answer": "Light spotting can be normal in early pregnancy, often due to implantation or cervical changes. However, any bleeding should be reported to your provider. Heavy bleeding, severe cramping, or tissue passage requires immediate medical attention.",
        
        "systolic_bp": 116,
        "diastolic_bp": 74,
        "heart_rate": 80,
        "weight_lbs": 138,
        "weight_gain_lbs": 1,
        "glucose_mg_dl": 84,
        "protein_urine": 0,
        "swelling_feet": 0,
        "headache": 0,
        "vision_blurry": 0,
        "bleeding_amount": 1,  # 0=none, 1=spotting, 2=light, 3=heavy
        "cramping": 3,  # 0-10 scale
        
        "pregnancy_week": 8,
        "trimester": 1,
        "days_pregnant": 56,
        "due_date_days_away": 224,
        
        "age": 27,
        "previous_pregnancies": 0,
        "bmi": 21.5,
        
        "urgency": "medium"
    },
    
    {
        "id": 8,
        "question": "What can I do about back pain at 32 weeks?",
        "answer": "Back pain is very common in the third trimester due to weight gain and postural changes. Try: prenatal yoga, pregnancy support belt, warm baths, proper posture, supportive shoes, and side sleeping with pillow support. Prenatal massage can also help.",
        
        "systolic_bp": 118,
        "diastolic_bp": 76,
        "heart_rate": 76,
        "weight_lbs": 170,
        "weight_gain_lbs": 28,
        "glucose_mg_dl": 92,
        "protein_urine": 0,
        "swelling_feet": 2,
        "headache": 0,
        "vision_blurry": 0,
        "back_pain": 7,  # 0-10 scale
        
        "pregnancy_week": 32,
        "trimester": 3,
        "days_pregnant": 224,
        "due_date_days_away": 56,
        
        "age": 31,
        "previous_pregnancies": 1,
        "bmi": 25.9,
        
        "urgency": "low"
    }
]


# ============================================================================
# SAVE DATA IN MULTIPLE FORMATS
# ============================================================================

def save_as_json():
    """Save as JSON (easiest to read and edit)"""
    filename = "pregnancy_qa_data.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(pregnancy_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved JSON: {filename}")
    return filename


def save_as_csv():
    """Save as CSV (for Excel/spreadsheets)"""
    filename = "pregnancy_qa_data.csv"
    
    # Get all possible field names
    all_fields = set()
    for item in pregnancy_data:
        all_fields.update(item.keys())
    
    fieldnames = sorted(all_fields)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pregnancy_data)
    
    print(f"✓ Saved CSV: {filename}")
    return filename


def display_summary():
    """Show what we created"""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal samples: {len(pregnancy_data)}")
    
    # Count by urgency
    urgency_counts = {}
    for item in pregnancy_data:
        urgency = item['urgency']
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    print(f"\nBy urgency level:")
    for urgency, count in sorted(urgency_counts.items()):
        print(f"  {urgency}: {count} questions")
    
    # Show sample
    print(f"\n" + "="*70)
    print("SAMPLE ENTRY #1")
    print("="*70)
    sample = pregnancy_data[0]
    print(f"\nQuestion: {sample['question']}")
    print(f"\nHealth Metrics:")
    print(f"  Blood Pressure: {sample['systolic_bp']}/{sample['diastolic_bp']}")
    print(f"  Heart Rate: {sample['heart_rate']}")
    print(f"  Week: {sample['pregnancy_week']}")
    print(f"\nAnswer: {sample['answer'][:100]}...")
    print(f"\nUrgency: {sample['urgency']}")


def create_template_for_adding_more():
    """Create an empty template so user can add their own data"""
    template = {
        "id": 9,
        "question": "YOUR QUESTION HERE",
        "answer": "YOUR ANSWER HERE",
        
        # Health metrics - fill in numbers
        "systolic_bp": 0,
        "diastolic_bp": 0,
        "heart_rate": 0,
        "weight_lbs": 0,
        "weight_gain_lbs": 0,
        "glucose_mg_dl": 0,
        "protein_urine": 0,
        "swelling_feet": 0,
        "headache": 0,
        "vision_blurry": 0,
        
        # Pregnancy info
        "pregnancy_week": 0,
        "trimester": 0,
        "days_pregnant": 0,
        "due_date_days_away": 0,
        
        # Patient info
        "age": 0,
        "previous_pregnancies": 0,
        "bmi": 0.0,
        
        "urgency": "low"  # low, medium, or high
    }
    
    filename = "pregnancy_qa_template.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump([template], f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Created template: {filename}")
    print("  (Copy this template to add more questions!)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Creating real pregnancy Q&A dataset...\n")
    
    # Save in both formats
    json_file = save_as_json()
    csv_file = save_as_csv()
    
    # Create template
    create_template_for_adding_more()
    
    # Show summary
    display_summary()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"""
You now have real pregnancy data in two formats:
  • {json_file} - Easy to read and edit
  • {csv_file} - Open in Excel

TO ADD MORE DATA:
1. Open pregnancy_qa_template.json
2. Copy the template
3. Fill in your own questions and health metrics
4. Add to pregnancy_qa_data.json

TO USE THIS DATA:
Next, we'll create a loader that reads this data
and feeds it to your attention model!
    """)


if __name__ == "__main__":
    main()