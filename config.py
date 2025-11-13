# --- Medical AI Configuration ---

import os
from typing import Dict, Any

# Environment-based configuration
ENV = os.getenv("ENVIRONMENT", "development")

# Base configuration
BASE_CONFIG = {
    # Memory and Context
    "memory_file": "memory.json",
    "max_memory_turns": 10,
    "memory_save_delay": 5.0,
    "response_cache_size": 100,
    
    # Audio Configuration
    "audio_sample_rate": 16000,
    "audio_duration": 5,
    "audio_chunk_duration": 0.1,
    "silence_threshold": 0.01,
    "silence_timeout": 1.5,
    
    # Model Configuration
    "whisper_model_size": "tiny",
    "tts_model": "tts_models/en/ljspeech/glow-tts",
    "llm_model": "gemma3:1b",
    "embedding_model": "ibm-granite/granite-embedding-small-english-r2",
    
    # RAG Configuration - UPDATED FOR LANGCHAIN + FAISS
    "vector_db_path": "./faiss_index",  # FAISS index directory
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_k": 4,
    "use_faiss": True,
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_top_k": 5,
    "similarity_score_threshold": 0.7,
    
    # Dataset Configuration - YOUR NEW DATASET (PDF)
    "dataset_name": "s41597-023-02406-6",
    "dataset_path": r"C:\Users\aviji\OneDrive\Desktop\whisper\Sydney2\Datasets\s41597-023-02406-6.pdf",
    "dataset_type": "pdf",  # pdf, json, or txt
    
    # Performance
    "max_workers": 3,
    "use_gpu": False,
    "model_compute_type": "int8",
    
    # Safety and Medical
    "medical_disclaimer": "⚕️ MEDICAL DISCLAIMER: I am an AI assistant and cannot provide medical diagnosis or treatment advice. For medical emergencies, contact emergency services immediately.",
    
    # Logging
    "log_level": "INFO",
    "log_file": "medical_ai.log"
}

# Development configuration
DEVELOPMENT_CONFIG = {
    **BASE_CONFIG,
    "whisper_model_size": "tiny",
    "model_compute_type": "int8",
    "max_workers": 2,
    "log_level": "DEBUG"
}

# Production configuration  
PRODUCTION_CONFIG = {
    **BASE_CONFIG,
    "whisper_model_size": "small",
    "model_compute_type": "float16",
    "max_workers": 4,
    "log_level": "WARNING",
    "use_response_caching": True
}

# Medical Emergency Configuration
EMERGENCY_CONFIG = {
    "keywords": [
        "chest pain", "heart attack", "stroke", "difficulty breathing", "severe pain",
        "unconscious", "bleeding", "overdose", "suicide", "emergency", "911", "ambulance",
        "severe headache", "high fever", "allergic reaction", "anaphylaxis", "seizure",
        "can't breathe", "choking", "severe bleeding", "poisoning"
    ],
    "priority_response": """🚨 MEDICAL EMERGENCY DETECTED 🚨

If this is a life-threatening emergency:
• Call 100 in  India , or your local emergency number IMMEDIATELY
• If unconscious/unresponsive: Call emergency services and start CPR if trained
• If having chest pain: Call emergency services, chew aspirin if not allergic, stay calm
• If stroke symptoms: Call 100 immediately - every minute counts

I am an AI and cannot provide emergency medical care. Professional medical help is essential for emergencies."""
}

# Medical Dataset Configuration - UPDATED
MEDICAL_DATASET = {
    "base_knowledge": [
        "Diabetes is a chronic disease affecting blood sugar levels. Symptoms include increased thirst, frequent urination, and fatigue. Type 1 diabetes is an autoimmune condition, while Type 2 is related to insulin resistance.",
        "Hypertension (high blood pressure) can cause headaches, shortness of breath, and chest pain. Regular monitoring is essential. Risk factors include obesity, high salt intake, stress, and genetics.",
        "Common cold symptoms include runny nose, sore throat, and mild fever. Rest and hydration are important. Most colds resolve within 7-10 days without treatment.",
        "Proper nutrition is essential for health. A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals. Aim for 5 servings of fruits and vegetables daily.",
        "Migraine headaches are characterized by intense throbbing pain, often on one side of the head. They may be accompanied by nausea, sensitivity to light, and visual disturbances.",
        "Asthma is a chronic respiratory condition causing wheezing, shortness of breath, and chest tightness. Triggers include allergens, exercise, and cold air. Inhalers provide relief.",
        "Depression is a mood disorder causing persistent sadness, loss of interest, and fatigue. It's treatable with therapy, medication, or both. Seek professional help if symptoms persist.",
        "Anxiety disorders involve excessive worry and fear. Physical symptoms include rapid heartbeat, sweating, and trembling. Cognitive behavioral therapy is often effective.",
        "Arthritis causes joint inflammation, pain, and stiffness. Osteoarthritis results from wear and tear, while rheumatoid arthritis is an autoimmune condition. Exercise and weight management help.",
        "Allergies occur when the immune system overreacts to substances like pollen, dust, or food. Symptoms include sneezing, itching, hives, and in severe cases, anaphylaxis.",
        "Influenza (flu) is a viral infection causing fever, body aches, cough, and fatigue. Annual vaccination is recommended. Antiviral medications can reduce severity if taken early.",
        "Pneumonia is a lung infection causing cough, fever, and difficulty breathing. It can be bacterial, viral, or fungal. Seek medical attention for persistent symptoms.",
        "Gastroesophageal reflux disease (GERD) causes heartburn and acid regurgitation. Avoid trigger foods, eat smaller meals, and don't lie down immediately after eating.",
        "Osteoporosis weakens bones, increasing fracture risk. Adequate calcium, vitamin D, and weight-bearing exercise are important for bone health. More common in older adults.",
        "Chronic kidney disease involves gradual loss of kidney function. Symptoms may include fatigue, swelling, and changes in urination. Diabetes and hypertension are major causes.",
        "Thyroid disorders affect metabolism. Hypothyroidism causes fatigue and weight gain, while hyperthyroidism causes weight loss and rapid heartbeat. Blood tests confirm diagnosis.",
        "Stroke occurs when blood flow to the brain is interrupted. Warning signs include sudden numbness, confusion, trouble speaking, and severe headache. Seek immediate emergency care.",
        "Heart disease includes conditions affecting the heart. Symptoms may include chest pain, shortness of breath, and fatigue. Risk factors include smoking, high cholesterol, and family history.",
        "Obesity increases risk for many health conditions including diabetes, heart disease, and joint problems. Healthy eating and regular physical activity support weight management.",
        "Insomnia is difficulty falling or staying asleep. Good sleep hygiene includes regular sleep schedule, limiting screen time before bed, and creating a comfortable sleep environment.",
        "Irritable bowel syndrome (IBS) causes abdominal pain, bloating, and changes in bowel habits. Stress management and dietary modifications can help manage symptoms.",
        "Celiac disease is an autoimmune disorder triggered by gluten. Symptoms include diarrhea, bloating, and fatigue. The only treatment is a strict gluten-free diet.",
        "Anemia is a deficiency of red blood cells or hemoglobin. Symptoms include fatigue, weakness, and pale skin. Iron deficiency is the most common cause.",
        "Skin cancer is the most common type of cancer. Warning signs include changes in moles or new skin growths. Regular skin checks and sun protection are important.",
        "Alzheimer's disease is a progressive neurological disorder causing memory loss and cognitive decline. Early symptoms include confusion and difficulty with familiar tasks.",
        "Parkinson's disease affects movement, causing tremors, stiffness, and balance problems. Symptoms worsen gradually over time. Medications and therapy can help manage symptoms.",
        "Chronic obstructive pulmonary disease (COPD) is a lung condition causing breathing difficulties. Smoking is the primary cause. Symptoms include chronic cough and shortness of breath.",
        "Urinary tract infections (UTIs) cause painful urination, frequent urge to urinate, and cloudy urine. More common in women. Antibiotics are typically prescribed for treatment.",
        "Eczema (atopic dermatitis) causes itchy, inflamed skin. Triggers include dry skin, irritants, and stress. Moisturizers and topical medications help control flare-ups.",
        "Psoriasis is an autoimmune condition causing scaly skin patches. It's chronic but manageable with topical treatments, light therapy, or systemic medications.",
        "Multiple sclerosis (MS) is an autoimmune disease affecting the central nervous system. Symptoms vary but may include fatigue, vision problems, and difficulty walking.",
        "Hepatitis is liver inflammation, often caused by viral infection. Hepatitis A, B, and C are most common. Vaccines are available for hepatitis A and B.",
        "Epilepsy is a neurological disorder causing recurrent seizures. Anti-seizure medications help control most cases. Avoid known triggers and get adequate sleep.",
        "Vitamin D deficiency is common, especially in areas with limited sunlight. It can cause bone pain and muscle weakness. Supplementation may be recommended.",
        "Dehydration occurs when fluid loss exceeds intake. Symptoms include thirst, dark urine, and dizziness. Drink water regularly, especially during exercise or hot weather.",
        "Food poisoning causes nausea, vomiting, and diarrhea. Usually resolves within 48 hours. Stay hydrated and seek medical care if symptoms are severe or prolonged.",
    ],
    
    "dataset_paths": {
        "primary": r"C:\Users\aviji\OneDrive\Desktop\whisper\Sydney2\Datasets\s41597-023-02406-6.json",
    }
}

def get_config() -> Dict[str, Any]:
    """Get configuration based on environment."""
    if ENV == "production":
        return PRODUCTION_CONFIG
    else:
        return DEVELOPMENT_CONFIG

# Export CONFIG for other modules to import
CONFIG = get_config()