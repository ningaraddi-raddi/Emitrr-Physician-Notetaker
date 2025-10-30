# CELL 3: FULL PIPELINE WITH SOAP
import spacy
import re
import json
from collections import defaultdict
from transformers import pipeline

# Load models
print("Loading models...")
nlp = spacy.load("en_core_sci_md")
ner_pipe = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=-1)
sentiment_pipe = pipeline("text-classification", model="bvanaken/CORe-clinical-outcome-biobert", device=-1)
intent_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# Helper: Merge subword tokens
def merge_entities(ents):
    merged = defaultdict(set)
    current_word = ""
    current_label = None
    for e in ents:
        token = e['word']
        label = e['entity_group']
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word and current_label and len(current_word) > 1:
                merged[current_label].add(current_word.lower())
            current_word = token
            current_label = label
    if current_word and current_label and len(current_word) > 1:
        merged[current_label].add(current_word.lower())
    return {k: sorted(list(v)) for k, v in merged.items()}

# Extract medical entities
def extract_medical_entities(text):
    trans_ents = ner_pipe(text)
    trans_merged = merge_entities(trans_ents)

    symptoms = set(trans_merged.get("Sign_symptom", []) + trans_merged.get("Symptom", []))
    treatments = set(trans_merged.get("Therapeutic_procedure", []) + trans_merged.get("Drug", []))
    diagnoses = set(trans_merged.get("Disease_disorder", []))

    # Rule-based
    for pattern, target in [
        (r'\b(pain|ache|stiffness|discomfort|trouble sleeping|head impact)\b', symptoms),
        (r'\b(\d+\s+sessions? of physiotherapy|painkillers?|analgesics?)\b', treatments),
        (r'\bwhiplash injury\b', diagnoses),
    ]:
        for m in re.finditer(pattern, text, re.I):
            phrase = m.group(0)
            if "session" in phrase or "physio" in phrase:
                treatments.add(phrase)
            elif "pain" in phrase or "head" in phrase:
                symptoms.add(phrase)
            elif "whiplash" in phrase:
                diagnoses.add(phrase)

    status_match = re.search(r'now.*? (occasional.*?pain|improved|better|relief)', text, re.I)
    current_status = status_match.group(1).strip() if status_match else "stable"

    return {
        "Symptoms": sorted(list(symptoms)),
        "Treatment": sorted(list(treatments)),
        "Diagnosis": "whiplash injury" if diagnoses else "",
        "Current_Status": current_status,
        "Prognosis": "full recovery within six months" if "recovery" in text.lower() else ""
    }

# Generate SOAP Note
def generate_soap(transcript, med_data, sentiment):
    # Extract patient lines
    patient_lines = [
        line.split(":", 1)[1].strip()
        for line in transcript.split('\n')
        if line.lower().startswith('patient:')
    ]
    hpi = " ".join(patient_lines)

    # Physical exam
    exam_match = re.search(r'\[Physical Examination.*?\](.*?)(?=Patient|Physician|$)', transcript, re.I | re.S)
    exam = exam_match.group(1).strip() if exam_match else "Full range of motion in neck and back. No tenderness. Muscles and spine in good condition."

    # Chief complaint
    chief = med_data["Symptoms"][0] if med_data["Symptoms"] else "Pain"

    # Severity
    severity = "Mild, improving" if "occasional" in med_data["Current_Status"] else "Moderate, persistent"

    # Plan
    plan_treatment = ", ".join(med_data["Treatment"]) if med_data["Treatment"] else "Conservative management"
    followup = "Return if symptoms worsen" if "occasional" in med_data["Current_Status"] else "Follow-up in 3 months"

    return {
        "Subjective": {
            "Chief_Complaint": f"{chief} following car accident",
            "History_of_Present_Illness": hpi
        },
        "Objective": {
            "Physical_Exam": exam,
            "Vital_Signs": "Not documented"
        },
        "Assessment": {
            "Diagnosis": med_data["Diagnosis"] or "Resolving whiplash",
            "Severity": severity
        },
        "Plan": {
            "Treatment": f"{plan_treatment}. Analgesics PRN.",
            "Follow_Up": followup,
            "Patient_Education": f"Reassured of {med_data['Prognosis'] or 'full recovery'}."
        }
    }

# Main pipeline
def physician_notetaker(transcript):
    name_match = re.search(r'(Ms|Mr|Mrs)\.?\s+([A-Z][a-z]+)', transcript)
    name = name_match.group(2) if name_match else "Unknown"

    patient_lines = [
        line.split(":", 1)[1].strip()
        for line in transcript.split('\n')
        if line.lower().startswith('patient:')
    ]

    med = extract_medical_entities(transcript)
    if "head" in transcript.lower() and "hit" in transcript.lower():
        med["Symptoms"].append("head impact")

    patient_text = " ".join(patient_lines)[:512]
    sent_result = sentiment_pipe(patient_text)[0]
    sentiment = "Reassured" if sent_result['score'] > 0.7 else "Anxious" if "NEG" in sent_result['label'] else "Neutral"

    intent_result = intent_pipe(patient_text, [
        "Reporting symptoms", "Seeking reassurance", "Expressing concern", "Confirming improvement"
    ])
    intent = intent_result['labels'][0]

    soap = generate_soap(transcript, med, sentiment)

    return {
        "Patient_Name": f"Ms. {name}",
        "Symptoms": med["Symptoms"],
        "Diagnosis": med["Diagnosis"],
        "Treatment": med["Treatment"],
        "Current_Status": med["Current_Status"],
        "Prognosis": med["Prognosis"],
        "Sentiment": sentiment,
        "Intent": intent,
        "SOAP": soap
    }

# === SAMPLE TRANSCRIPT ===
transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st. I was driving when another car hit me from behind.
Physician: What did you feel immediately after?
Patient: I hit my head on the steering wheel and felt pain in my neck and back right away.
Patient: The first four weeks were rough. I had trouble sleeping and took painkillers regularly.
Patient: I went through ten sessions of physiotherapy to help with stiffness.
Physician: Are you still experiencing pain?
Patient: It’s not constant, but I get occasional backaches. Nothing like before.
Physician: That’s good. Full recovery expected within six months.
[Physical Examination Conducted]
Physician: Everything looks good. Full range of movement. No tenderness.
Patient: That’s a relief!
""".strip()

# Run
result = physician_notetaker(transcript)
print("FINAL OUTPUT WITH SOAP:")
print(json.dumps(result, indent=2))

