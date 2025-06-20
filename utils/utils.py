import re
from difflib import get_close_matches

# Clean input text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Disease/Symptom-to-Solution Mapping
def get_disease_solution_map():
    return {
        # General outcomes
        "positive": "AI believes symptoms are not severe.",
        "negative": "AI believes symptoms could be severe. Please consult a doctor.",
        "unknown": "AI could not determine the condition. Please provide more details or consult a professional.",
        "normal": "Your symptoms appear normal. Stay healthy and maintain a balanced lifestyle.",

        # Infectious diseases
        "flu": "Stay hydrated, rest, and take antivirals if prescribed.",
        "influenza": "Stay hydrated, rest, and take antivirals if prescribed.",
        "covid-19": "Isolate, monitor oxygen levels, and seek help if needed.",
        "coronavirus": "Isolate, monitor oxygen levels, and seek help if needed.",
        "cold": "Get rest, stay warm, and drink plenty of fluids.",
        "viral infection": "Rest, hydrate, and take medication if prescribed.",
        "dengue": "Get tested, stay hydrated, and consult a doctor immediately.",
        "malaria": "Use antimalarial drugs as prescribed and avoid mosquito exposure.",
        "tuberculosis": "Seek immediate medical attention and follow the full course of antibiotics.",
        "typhoid": "Maintain hygiene, stay hydrated, and follow the prescribed antibiotics course.",
        "hepatitis": "Avoid alcohol, rest, and follow doctor’s dietary guidelines.",
        "ebola": "Seek emergency care. Isolate to prevent spread.",
        "zika": "Rest, fluids, and acetaminophen for pain relief.",

        # Chronic diseases
        "diabetes": "Maintain a healthy diet, exercise, and monitor sugar levels.",
        "type 1 diabetes": "Insulin management and regular glucose monitoring are essential.",
        "type 2 diabetes": "Diet, exercise, and medication adherence are key.",
        "hypertension": "Reduce salt, manage stress, and follow medication.",
        "high blood pressure": "Reduce salt, manage stress, and follow medication.",
        "asthma": "Avoid allergens and use prescribed inhalers.",
        "chronic bronchitis": "Quit smoking, avoid pollutants, and follow treatment.",
        "copd": "Avoid smoking, use inhalers, and seek regular checkups.",
        "heart disease": "Exercise moderately and follow cardiac diet.",
        "cardiovascular disease": "Exercise moderately and follow cardiac diet.",
        "stroke": "Immediate medical intervention is critical. Rehabilitate under guidance.",
        "arthritis": "Exercise gently, use anti-inflammatory medication, and maintain joint care.",
        "osteoporosis": "Calcium/vitamin D intake and weight-bearing exercises help.",
        "epilepsy": "Use antiepileptic drugs regularly and avoid triggers.",

        # Common symptoms
        "fever": "Monitor temperature, drink fluids, and rest. Seek help if fever persists.",
        "headache": "Rest, stay hydrated, and avoid screen time. Seek help if frequent.",
        "cough": "Stay hydrated, use cough syrup, and consult if persistent.",
        "sore throat": "Gargle with salt water, stay hydrated, and rest.",
        "vomiting": "Avoid solid food initially, sip fluids slowly, and rest.",
        "diarrhea": "Stay hydrated and avoid spicy or oily food.",
        "body ache": "Rest and take over-the-counter pain relief if necessary.",
        "fatigue": "Get adequate sleep, eat well, and manage stress.",
        "dizziness": "Sit or lie down, hydrate, and consult if recurring.",
        "nausea": "Eat light food, hydrate, and rest.",
        "chills": "Keep warm and monitor for fever or infections.",
        "shortness of breath": "Seek immediate medical attention especially if sudden or severe.",
        "chest pain": "This may be serious — consult a doctor immediately.",
        "runny nose": "Use a nasal spray and stay hydrated.",
        "congestion": "Inhale steam, rest, and use decongestants if needed.",
        "loss of taste": "Monitor symptoms and consult a doctor if it persists.",
        "loss of smell": "Could be due to infection — monitor and seek help.",
        "itchy eyes": "Use antihistamine drops and avoid allergens.",

        # Mental health
        "anxiety": "Practice relaxation, avoid caffeine, and consider therapy.",
        "depression": "Talk to a mental health professional, stay socially connected, and get regular exercise.",
        "stress": "Take breaks, meditate, and manage workload effectively.",
        "panic attack": "Practice breathing exercises and seek professional help if frequent.",
        "bipolar disorder": "Medication and therapy are crucial for stability.",

        # Women's health
        "menstrual cramps": "Use heat packs, stay hydrated, and take pain relievers if needed.",
        "pcos": "Maintain a healthy weight and diet, and take hormonal treatment if prescribed.",
        "pregnancy": "Ensure regular checkups, eat balanced meals, and avoid strenuous activity.",
        "menopause": "Stay active, eat well, and consult for hormone therapy if needed.",

        # Skin and allergies
        "allergy": "Avoid allergens and use antihistamines as prescribed.",
        "eczema": "Use moisturizers and avoid irritants.",
        "acne": "Maintain face hygiene and avoid greasy food.",
        "rash": "Avoid scratching, keep area clean, and use medicated creams if needed.",
        "hives": "Identify triggers and take antihistamines.",
        "psoriasis": "Use prescribed creams and avoid known irritants.",

        # Digestive issues
        "indigestion": "Avoid overeating, spicy food, and eat slowly.",
        "gastritis": "Avoid NSAIDs, spicy foods, and consult a gastroenterologist if persistent.",
        "ulcer": "Follow a bland diet and prescribed medication regularly.",
        "constipation": "Increase fiber intake and drink more water.",
        "bloating": "Eat smaller meals and avoid gas-producing foods.",
        "acid reflux": "Avoid lying down after meals and avoid spicy/fatty food.",

        # Urinary conditions
        "uti": "Drink lots of water and complete any antibiotic course prescribed.",
        "urinary tract infection": "Drink water and avoid holding urine for too long.",
        "kidney stones": "Drink water and follow prescribed treatment for pain and expulsion.",

        # Pediatric/geriatric
        "measles": "Isolate, rest, and follow doctor’s instructions.",
        "chickenpox": "Rest, avoid scratching, and follow antiviral treatment if needed.",
        "alzheimer's": "Medication and routine-based care help manage symptoms.",
        "parkinson's": "Physiotherapy and medication can help manage symptoms.",
        "autism": "Early therapy and structured support can help with development.",

        # Miscellaneous
        "anemia": "Consume iron-rich foods and take supplements if prescribed.",
        "thyroid": "Follow medication and check TSH levels regularly.",
        "migraine": "Avoid triggers, stay hydrated, and use prescribed medication.",
        "dehydration": "Drink oral rehydration solutions or water regularly.",
        "insomnia": "Avoid caffeine at night, maintain sleep hygiene, and reduce screen time.",
        "headache and acne": "Avoid oily or sugary foods, drink plenty of water, and keep your face clean with non-comedogenic products. Also, manage stress and maintain a regular sleep schedule."
    }

# Get description for a given label
def lookup_description(label):
    return get_disease_solution_map().get(label.lower(), "No medical advice available for this condition.")

# Main enrichment function: combines input matching and prediction labels
def enrich_prediction(prediction, input_text=None):
    enriched = []
    seen = set()
    disease_map = get_disease_solution_map()

    # Step 1: Keyword extraction from user input
    if input_text:
        input_clean = clean_text(input_text)
        for key in sorted(disease_map.keys(), key=len, reverse=True):
            if key in input_clean and key not in seen:
                seen.add(key)
                enriched.append({
                    "label": key.upper(),
                    "score": 0.90,
                    "description": disease_map[key]
                })

    # Step 2: Add model predictions (if not already matched)
    if isinstance(prediction, str):
        label = prediction.lower()
        if label not in seen:
            seen.add(label)
            enriched.append({
                "label": label.upper(),
                "description": lookup_description(label)
            })

    elif isinstance(prediction, list):
        for item in prediction:
            label = (item.get("label") or item.get("labels") or "").lower()
            score = item.get("score", None)
            if label and label not in seen:
                seen.add(label)
                enriched.append({
                    "label": label.upper(),
                    "score": score,
                    "description": lookup_description(label)
                })

    return enriched
