# src/soap_generator_auto_clean.py
import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DATA_DIR = "data/processed"
NER_RESULTS_FILE = os.path.join(DATA_DIR, "ner_results.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "summary.json")
SENTIMENT_FILE = os.path.join(DATA_DIR, "sentiment_results.json")
SOAP_OUTPUT = os.path.join(DATA_DIR, "soap_note_auto.json")

# ----------------------------------------------------------------------
# 1. Load all data
# ----------------------------------------------------------------------
def load_data():
    for file in [NER_RESULTS_FILE, SUMMARY_FILE, SENTIMENT_FILE]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} not found!")

    with open(NER_RESULTS_FILE, "r", encoding="utf-8") as f:
        ner_data = json.load(f)
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(SENTIMENT_FILE, "r", encoding="utf-8") as f:
        sentiment_data = json.load(f)

    return ner_data, summary, sentiment_data

# ----------------------------------------------------------------------
# 2. Map sentiment/intent to severity
# ----------------------------------------------------------------------
def infer_severity(sentiment, intent):
    severity = "Mild"
    if sentiment == "Anxious":
        severity = "Moderate to High"
    elif sentiment == "Neutral":
        severity = "Mild"
    elif sentiment == "Reassured":
        severity = "Mild/Improving"

    if intent in ["Seeking reassurance", "Expressing concern"]:
        if severity in ["Mild", "Mild/Improving"]:
            severity = "Mild/Moderate"
        elif severity == "Moderate to High":
            severity = "High"
    return severity

# ----------------------------------------------------------------------
# 3. Deduplicate symptoms and clean text
# ----------------------------------------------------------------------
def clean_text_list(items):
    # Remove duplicates, strip whitespace, capitalize first letters
    return [i.strip().capitalize() for i in sorted(set(items)) if i.strip()]

# ----------------------------------------------------------------------
# 4. Map to SOAP Sections
# ----------------------------------------------------------------------
def map_to_soap(ner_data, summary, sentiment_data):
    patient_ents = ner_data.get("Patient", {})

    # Subjective
    symptoms = clean_text_list(patient_ents.get("Symptoms", []))
    subjective = {
        "Chief_Complaint": ", ".join(symptoms),
        "History_of_Present_Illness": summary.get("Patient_Text", "")
    }

    # Objective
    measurements = patient_ents.get("Measurements", [])
    objective_text = []
    if measurements:
        objective_text.append(f"Patient measurements: {', '.join(measurements)}.")
    if patient_ents.get("Treatment"):
        objective_text.append(f"Treatment received: {', '.join(patient_ents.get('Treatment', []))}.")
    objective = {
        "Physical_Exam": " ".join(objective_text) if objective_text else "No structured exam data available.",
        "Observations": "Patient appears stable."
    }

    # Assessment
    diagnosis = clean_text_list(patient_ents.get("Diagnosis", []))
    assessment = {
        "Diagnosis": ", ".join(diagnosis) if diagnosis else summary.get("Diagnosis", "Not specified"),
        "Severity": infer_severity(sentiment_data.get("Sentiment", "Neutral"),
                                   sentiment_data.get("Intent", "Reporting symptoms"))
    }

    # Plan
    treatment = clean_text_list(patient_ents.get("Treatment", []))
    plan = {
        "Treatment": ", ".join(treatment) if treatment else "No treatment recorded",
        "Follow-Up": "Patient to follow up as needed or if symptoms worsen."
    }

    return {
        "Subjective": subjective,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }

# ----------------------------------------------------------------------
# 5. Refine text with LLM (compact & readable)
# ----------------------------------------------------------------------
def refine_text_with_llm(soap_note):
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    refined_soap = {}
    for section, content in soap_note.items():
        refined_soap[section] = {}
        for key, text in content.items():
            if not text:
                refined_soap[section][key] = text
                continue
            prompt = f"Rephrase the following medical note concisely for clinicians:\n{text}"
            output = generator(prompt, max_length=150, do_sample=False)[0]["generated_text"]

            # Clean redundant instructions from model output
            for phrase in ["Keep it concise", "Write a medical note"]:
                output = output.replace(phrase, "")
            refined_soap[section][key] = output.strip()
    return refined_soap

# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    ner_data, summary, sentiment_data = load_data()

    if "Patient_Text" not in summary:
        summary["Patient_Text"] = " ".join(ner_data.get("Patient", {}).get("Symptoms", []))

    soap_note = map_to_soap(ner_data, summary, sentiment_data)
    refined_soap = refine_text_with_llm(soap_note)

    # Save final SOAP note
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SOAP_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(refined_soap, f, indent=4, ensure_ascii=False)

    print("[SUCCESS] Cleaned SOAP note generated and saved to:", SOAP_OUTPUT)
    print(json.dumps(refined_soap, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
