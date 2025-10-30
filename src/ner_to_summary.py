# # src/ner_to_summary.py
# import os
# import json
# import re

# DATA_DIR = "data/processed"
# NER_RESULTS_FILE = os.path.join(DATA_DIR, "ner_results.json")


# def clean_entities(entities_list, min_len=3, remove_fragments=True):
#     """
#     Cleans a list of entities:
#     - Removes short fragments (< min_len chars)
#     - Deduplicates ignoring case
#     - Keeps longest overlapping phrases
#     - Capitalizes first letters
#     - Removes obvious partial/ambiguous words
#     """
#     if not entities_list:
#         return []

#     # Remove short fragments
#     entities_list = [e.strip() for e in entities_list if len(e.strip()) >= min_len]

#     # Remove obvious ambiguous fragments
#     ambiguous_words = {"recover", "stiff", "pain"}  # 'pain' can be merged elsewhere
#     entities_list = [e for e in entities_list if e.lower() not in ambiguous_words]

#     # Deduplicate ignoring case and keep longest version
#     temp_set = {}
#     for e in entities_list:
#         key = e.lower()
#         if key in temp_set:
#             if len(e) > len(temp_set[key]):
#                 temp_set[key] = e
#         else:
#             temp_set[key] = e

#     # Remove substrings of longer entities
#     final_set = set(temp_set.values())
#     for e1 in temp_set.values():
#         for e2 in temp_set.values():
#             if e1 != e2 and e1.lower() in e2.lower():
#                 final_set.discard(e1)
#                 break

#     # Capitalize first letters
#     cleaned = [e.title() for e in final_set]
#     return sorted(cleaned)


# def extract_patient_name(text):
#     """
#     Try to extract patient name from patient text using regex or fallback to NER
#     """
#     if not text:
#         return "Unknown"

#     # Common patterns
#     patterns = [
#         r"my name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
#         r"i am ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
#         r"this is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
#     ]
#     for pat in patterns:
#         m = re.search(pat, text, re.I)
#         if m:
#             return m.group(1).title()

#     # Fallback: use NER 'PERSON' if available
#     return "Unknown"


# def infer_current_status(patient_text):
#     """
#     Extract phrases indicating current status like:
#     "I am still feeling...", "I still have...", "Currently..."
#     """
#     status_patterns = [
#         r"i am still feeling ([a-zA-Z\s]+?)(?:\.|,|$)",
#         r"i still have ([a-zA-Z\s]+?)(?:\.|,|$)",
#         r"currently ([a-zA-Z\s]+?)(?:\.|,|$)"
#     ]
#     statuses = []
#     for pat in status_patterns:
#         for m in re.finditer(pat, patient_text, re.I):
#             phrase = m.group(1).strip()
#             if len(phrase) > 2:
#                 statuses.append(phrase.title())
#     return ", ".join(statuses)


# def generate_summary(ner_data):
#     patient_text = ner_data.get("Patient_Text", "")
#     patient_ents = ner_data.get("Patient", {})

#     # Clean entities
#     symptoms = clean_entities(patient_ents.get("Symptoms", []))
#     diagnosis_list = clean_entities(patient_ents.get("Diagnosis", []))
#     treatment = clean_entities(patient_ents.get("Treatment", []))
#     prognosis_list = clean_entities(patient_ents.get("Prognosis", []))

#     # Pick main diagnosis and prognosis
#     main_diag = diagnosis_list[0] if diagnosis_list else ""
#     main_prognosis = prognosis_list[0] if prognosis_list else ""

#     # Current status inference
#     current_status = infer_current_status(patient_text)

#     summary = {
#         "Patient_Name": extract_patient_name(patient_text),
#         "Symptoms": symptoms,
#         "Diagnosis": main_diag,
#         "Treatment": treatment,
#         "Current_Status": current_status,
#         "Prognosis": main_prognosis,
#     }
#     return summary


# def main():
#     if not os.path.exists(NER_RESULTS_FILE):
#         print(f"[ERROR] NER results not found at {NER_RESULTS_FILE}")
#         return

#     with open(NER_RESULTS_FILE, "r", encoding="utf-8") as f:
#         ner_data = json.load(f)

#     # Load patient text
#     patient_file = os.path.join(DATA_DIR, "patient_text.txt")
#     patient_text = ""
#     if os.path.exists(patient_file):
#         with open(patient_file, "r", encoding="utf-8") as f:
#             patient_text = f.read()
#     ner_data["Patient_Text"] = patient_text

#     summary = generate_summary(ner_data)

#     # Save summary
#     out_file = os.path.join(DATA_DIR, "summary.json")
#     with open(out_file, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=4, ensure_ascii=False)

#     print("[SUCCESS] Summary generated:")
#     print(json.dumps(summary, indent=4))


# if __name__ == "__main__":
#     main()















# src/ner_to_summary.py
import os
import json
import re
from collections import defaultdict

DATA_DIR = "data/processed"
NER_RESULTS_FILE = os.path.join(DATA_DIR, "ner_results.json")

def clean_entities(entities_list):
    """Clean list: remove short fragments, dedupe, keep longest overlapping phrases, title-case."""
    entities_list = [e.strip() for e in entities_list if len(e.strip()) >= 3]
    temp_set = {}
    for e in entities_list:
        key = e.lower()
        if key in temp_set:
            if len(e) > len(temp_set[key]):
                temp_set[key] = e
        else:
            temp_set[key] = e
    final_set = set(temp_set.values())
    for e1 in temp_set.values():
        for e2 in temp_set.values():
            if e1 != e2 and e1.lower() in e2.lower():
                final_set.discard(e1)
                break
    return sorted([e.title() for e in final_set])

def extract_patient_name(text):
    """Extract patient name using simple regex."""
    patterns = [
        r"my name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"i am ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).title()
    return "Unknown"

def extract_current_status(text):
    """Extract phrases describing current status of patient."""
    patterns = [
        r'i still (?:feel|have|experience) ([a-z\s]+?)(?:\.|,|$)',
        r'sometimes i (?:feel|have) ([a-z\s]+?)(?:\.|,|$)',
        r'occasional(?:ly)? ([a-z\s]+?)(?:\.|,|$)'
    ]
    statuses = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            phrase = m.group(1).strip()
            if len(phrase) > 2:
                statuses.append(phrase)
    return clean_entities(statuses)

def generate_summary(ner_data):
    patient_text = ner_data.get("Patient_Text", "")
    patient_ents = ner_data.get("Patient", {})
    doctor_ents = ner_data.get("Doctor", {})

    # Clean entities
    symptoms = clean_entities(patient_ents.get("Symptoms", []))
    diagnosis = clean_entities(patient_ents.get("Diagnosis", []))
    treatment = clean_entities(patient_ents.get("Treatment", []))
    prognosis = clean_entities(patient_ents.get("Prognosis", []))

    # Fill missing diagnosis/prognosis from doctor
    if not diagnosis:
        diagnosis = clean_entities(doctor_ents.get("Diagnosis", []))
    if not prognosis:
        prognosis = clean_entities(doctor_ents.get("Prognosis", []))

    # Current status extraction
    current_status = extract_current_status(patient_text)

    summary = {
        "Patient_Name": extract_patient_name(patient_text),
        "Symptoms": symptoms,
        "Diagnosis": diagnosis[0] if diagnosis else "",
        "Treatment": treatment,
        "Current_Status": current_status[0] if current_status else "",
        "Prognosis": prognosis[0] if prognosis else "",
    }
    return summary

def main():
    if not os.path.exists(NER_RESULTS_FILE):
        print(f"[ERROR] NER results not found at {NER_RESULTS_FILE}")
        return

    with open(NER_RESULTS_FILE, "r", encoding="utf-8") as f:
        ner_data = json.load(f)

    # Include patient text for name/current_status extraction
    patient_text_file = os.path.join(DATA_DIR, "patient_text.txt")
    if os.path.exists(patient_text_file):
        ner_data["Patient_Text"] = open(patient_text_file, "r", encoding="utf-8").read()
    else:
        ner_data["Patient_Text"] = ""

    summary = generate_summary(ner_data)

    # Save summary
    out_file = os.path.join(DATA_DIR, "summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("[SUCCESS] Summary generated:")
    print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    main()
