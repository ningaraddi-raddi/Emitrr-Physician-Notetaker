






# src/ner_extractor.py
import os
import json
import re
import spacy
from collections import defaultdict
from spacy.matcher import Matcher, PhraseMatcher
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. MEDICAL KNOWLEDGE BASE
# ----------------------------------------------------------------------
MEDICAL_KNOWLEDGE = {
    "symptoms": [
        "pain", "ache", "discomfort", "soreness", "hurt", "stiffness", "swelling",
        "nausea", "vomiting", "dizziness", "fatigue", "weakness", "fever", "cough",
        "headache", "migraine", "numbness", "tingling", "burning", "itching",
        "blurred vision", "double vision", "ringing", "bleeding", "bruising",
        "shortness of breath", "chest pain", "palpitations", "anxiety", "depression",
        "insomnia", "trouble sleeping", "loss of appetite", "weight loss", "weight gain",
        "confusion", "memory loss", "tremor", "seizure", "paralysis", "cramping"
    ],
    "body_parts": [
        "head", "neck", "back", "spine", "shoulder", "arm", "elbow", "wrist", "hand",
        "chest", "abdomen", "stomach", "hip", "leg", "knee", "ankle", "foot",
        "brain", "heart", "lung", "liver", "kidney", "eye", "ear", "throat",
        "cervical", "lumbar", "thoracic"
    ],
    "treatments": [
        "physiotherapy", "therapy", "physical therapy", "occupational therapy",
        "medication", "medicine", "drug", "painkiller", "analgesic", "antibiotic",
        "surgery", "operation", "procedure", "examination", "x-ray", "mri", "ct scan",
        "ultrasound", "blood test", "biopsy", "injection", "infusion",
        "rest", "ice", "heat", "compression", "elevation", "exercise",
        "counseling", "rehabilitation", "insulin", "prescription"
    ],
    "diagnoses": [
        "injury", "fracture", "break", "sprain", "strain", "tear", "rupture",
        "whiplash", "concussion", "trauma", "wound", "laceration", "contusion",
        "infection", "inflammation", "disease", "disorder", "syndrome", "condition",
        "diabetes", "hypertension", "arthritis", "cancer", "tumor", "mass",
        "pneumonia", "bronchitis", "asthma", "copd", "heart disease", "stroke"
    ],
    "prognosis_terms": [
        "recovery", "healing", "prognosis", "outlook", "improvement", "progress",
        "full recovery", "partial recovery", "permanent", "temporary", "chronic",
        "acute", "long-term", "short-term", "stable", "improving", "worsening",
        "expected to recover", "no long-term effects", "complete healing"
    ],
    "time_indicators": [
        "weeks", "months", "years", "days", "hours", "session", "visit", "appointment"
    ]
}

# ----------------------------------------------------------------------
# 2. FILE I/O
# ----------------------------------------------------------------------
def load_transcript(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"[WARN] {name} not found")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ----------------------------------------------------------------------
# 3. MODEL LOADING
# ----------------------------------------------------------------------
def load_transformer_ner() -> pipeline:
    model_name = "d4data/biomedical-ner-all"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner = pipeline("ner", model=model, tokenizer=tokenizer,
                       aggregation_strategy="simple", device=-1)
        print(f"[OK] Transformer NER loaded: {model_name}")
        return ner
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_name}: {e}")


def load_spacy() -> spacy.Language:
    candidates = ["en_core_sci_md", "en_core_sci_sm", "en_core_web_sm"]
    for name in candidates:
        if spacy.util.is_package(name):
            nlp = spacy.load(name)
            print(f"[OK] Loaded spaCy model: {name}")
            return nlp
    # Install fallback
    print("[INFO] Installing en_core_web_sm...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")
    print("[OK] Loaded fallback model: en_core_web_sm")
    return nlp


# ----------------------------------------------------------------------
# 4. MATCHERS
# ----------------------------------------------------------------------
def build_matchers(nlp: spacy.Language):
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # SYMPTOM
    symptom_patterns = [
        [{"LOWER": {"IN": MEDICAL_KNOWLEDGE["body_parts"]}},
         {"LOWER": {"IN": ["pain", "ache", "discomfort", "soreness", "hurt", "stiffness"]}}],
        [{"LOWER": {"IN": ["pain", "ache", "discomfort"]}}, {"LOWER": "in"},
         {"LOWER": {"IN": MEDICAL_KNOWLEDGE["body_parts"]}}],
        [{"LOWER": {"IN": MEDICAL_KNOWLEDGE["symptoms"]}}],
    ]
    matcher.add("SYMPTOM", symptom_patterns)

    # TREATMENT
    treatment_patterns = [
        [{"IS_DIGIT": True},
         {"LOWER": {"IN": ["session", "sessions", "treatment", "treatments"]}},
         {"LOWER": "of", "OP": "?"},
         {"LOWER": {"IN": MEDICAL_KNOWLEDGE["treatments"]}, "OP": "?"}],
        [{"LOWER": {"IN": MEDICAL_KNOWLEDGE["treatments"]}}],
    ]
    matcher.add("TREATMENT", treatment_patterns)

    # DIAGNOSIS
    diag_patterns = [
        [{"LOWER": {"IN": MEDICAL_KNOWLEDGE["diagnoses"]}}],
        [{"LOWER": {"IN": MEDICAL_KNOWLEDGE["body_parts"]}},
         {"LOWER": {"IN": MEDICAL_KNOWLEDGE["diagnoses"]}}],
    ]
    matcher.add("DIAGNOSIS", diag_patterns)

    # PROGNOSIS
    prog_patterns = [
        [{"LOWER": {"IN": MEDICAL_KNOWLEDGE["prognosis_terms"]}}],
        [{"LOWER": "full"}, {"LOWER": "recovery"}],
        [{"LOWER": "no"}, {"LOWER": {"IN": ["long-term", "permanent"]}}, {"IS_ALPHA": True}],
    ]
    matcher.add("PROGNOSIS", prog_patterns)

    # PHRASE MATCHER
    symptom_phrases = [nlp.make_doc(p) for p in [
        "neck pain", "back pain", "head pain", "chest pain",
        "trouble sleeping", "difficulty breathing", "blurred vision",
        "shortness of breath", "loss of appetite"
    ]]
    treatment_phrases = [nlp.make_doc(p) for p in [
        "physiotherapy sessions", "physical therapy", "pain medication",
        "x-ray", "ct scan", "mri scan", "blood test"
    ]]
    diag_phrases = [nlp.make_doc(p) for p in [
        "whiplash injury", "car accident", "type 2 diabetes"
    ]]

    phrase_matcher.add("SYMPTOM", symptom_phrases)
    phrase_matcher.add("TREATMENT", treatment_phrases)
    phrase_matcher.add("DIAGNOSIS", diag_phrases)

    return matcher, phrase_matcher


# ----------------------------------------------------------------------
# 5. EXTRACTION HELPERS
# ----------------------------------------------------------------------
def transformer_entities(text: str, ner_pipe) -> dict:
    try:
        raw = ner_pipe(text)
        out = defaultdict(set)
        for e in raw:
            word = e["word"].replace("##", "").strip()
            if len(word) > 1:
                out[e["entity_group"]].add(word)
        return {k: list(v) for k, v in out.items()}
    except Exception as e:
        print(f"[WARN] Transformer failed: {e}")
        return {}


def spacy_entities(doc) -> dict:
    out = defaultdict(set)
    for ent in doc.ents:
        out[ent.label_].add(ent.text.strip())
    return {k: list(v) for k, v in out.items()}


def pattern_entities(doc, matcher, phrase_matcher, nlp) -> dict:
    out = defaultdict(set)

    # Token matcher
    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]
        out[label].add(doc[start:end].text.strip())

    # Phrase matcher
    for match_id, start, end in phrase_matcher(doc):
        label = nlp.vocab.strings[match_id]
        out[label].add(doc[start:end].text.strip())

    return {k: list(v) for k, v in out.items()}


def regex_entities(text: str) -> dict:
    out = defaultdict(list)
    out["MEASUREMENT"].extend(re.findall(r'\d+\s*(?:sessions?|weeks?|months?|days?|mg/dl|mmol/l)', text, re.I))
    out["TIME_REFERENCE"].extend(re.findall(r'(?:for|since|about)\s+(\d+\s+(?:weeks?|months?|days?))', text, re.I))
    return dict(out)


# ----------------------------------------------------------------------
# 6. MAPPING
# ----------------------------------------------------------------------
CATEGORY_MAP = {
    "SYMPTOM": "Symptoms",
    "SIGN_SYMPTOM": "Symptoms",
    "PROBLEM": "Symptoms",
    "DISEASE": "Diagnosis",
    "DISEASE_DISORDER": "Diagnosis",
    "DIAGNOSIS": "Diagnosis",
    "CONDITION": "Diagnosis",
    "TREATMENT": "Treatment",
    "THERAPEUTIC_OR_PREVENTIVE_PROCEDURE": "Treatment",
    "MEDICATION": "Treatment",
    "DRUG": "Treatment",
    "DIAGNOSTIC_PROCEDURE": "Treatment",
    "TEST": "Treatment",
    "PROGNOSIS": "Prognosis",
    "MEASUREMENT": "Measurements",
    "TIME_REFERENCE": "Measurements",
    "ANATOMY": "Symptoms",
    "BODY_PART": "Symptoms",
    "PERSON": "Patient_Info",
    "DATE": "Measurements",
    "TIME": "Measurements",
    # Rule-based
    "SYMPTOM": "Symptoms",
    "TREATMENT": "Treatment",
    "DIAGNOSIS": "Diagnosis",
    "PROGNOSIS": "Prognosis",
}


def map_to_categories(entities: dict) -> dict:
    cats = {cat: set() for cat in ["Symptoms", "Diagnosis", "Treatment", "Prognosis", "Patient_Info", "Measurements"]}
    for label, words in entities.items():
        target = CATEGORY_MAP.get(label.upper())
        if target and words:
            cats[target].update(w.lower() for w in words if w.strip())
    return {k: sorted(list(v)) for k, v in cats.items()}


# ----------------------------------------------------------------------
# 7. CONTEXT REFINEMENT (Patient only)
# ----------------------------------------------------------------------
def refine_patient_context(text: str, cats: dict) -> dict:
    lower = text.lower()

    # Symptoms
    for pat in [
        r'(?:i\s+have|i\'m\s+having|feeling|having)\s+([a-z\s]+?)(?:\.|,|and|$)',
        r'(?:my\s+)([a-z]+)\s+(?:is|are|was|were)\s+(?:painful|sore|hurting|aching)',
        r'(?:pain|ache|discomfort)\s+in\s+(?:my\s+)?([a-z]+)',
    ]:
        for m in re.finditer(pat, lower):
            phrase = m.group(1).strip()
            if len(phrase) > 2:
                cats["Symptoms"].append(phrase)

    # Treatments
    for pat in [
        r'(?:i\s+(?:took|received|had|went\s+through))\s+([a-z\s\d]+?)(?:\.|,|for|$)',
        r'(\d+)\s+(?:sessions?|treatments?)\s+of\s+([a-z\s]+)',
    ]:
        for m in re.finditer(pat, lower):
            phrase = " ".join(filter(None, m.groups())).strip()
            if len(phrase) > 2:
                cats["Treatment"].append(phrase)

    # Diagnosis (patient repeating)
    for pat in [r'(?:they\s+said|told\s+me)\s+(?:i\s+have|a)\s+([a-z\s]+?)(?:\.|,|and|$)']:
        for m in re.finditer(pat, lower):
            phrase = m.group(1).strip()
            if len(phrase) > 2:
                cats["Diagnosis"].append(phrase)

    # Prognosis
    for pat in [r'(?:doctor\s+said|they\s+told\s+me)\s+(full\s+recovery|no\s+long-term\s+damage)']:
        for m in re.finditer(pat, lower):
            cats["Prognosis"].append(m.group(1).strip())

    # Dedupe
    for k in cats:
        cats[k] = sorted(list({w.strip() for w in cats[k] if len(w.strip()) > 1}))
    return cats


# ----------------------------------------------------------------------
# 8. MAIN EXTRACTION
# ----------------------------------------------------------------------
def extract_for_speaker(text: str, ner_pipe, nlp, matcher, phrase_matcher, is_patient: bool):
    if not text:
        return {cat: [] for cat in ["Symptoms", "Diagnosis", "Treatment", "Prognosis", "Patient_Info", "Measurements"]}

    # 1. Transformer
    trans = transformer_entities(text, ner_pipe)

    # 2. spaCy NER
    doc = nlp(text)
    spacy_out = spacy_entities(doc)

    # 3. Pattern matching (PASS NLP!)
    pattern_out = pattern_entities(doc, matcher, phrase_matcher, nlp)

    # 4. Regex
    regex_out = regex_entities(text)

    # Merge
    merged = defaultdict(list)
    for src in (trans, spacy_out, pattern_out, regex_out):
        for label, items in src.items():
            merged[label].extend(items)

    # Dedupe
    for label in merged:
        merged[label] = list({w.strip() for w in merged[label] if w.strip()})

    # Map
    cats = map_to_categories(merged)

    # Refine (only patient)
    if is_patient:
        cats = refine_patient_context(text, cats)

    return cats


# ----------------------------------------------------------------------
# 9. RUNNER
# ----------------------------------------------------------------------
def run_ner_extraction():
    print("\n=== MEDICAL NER PIPELINE ===\n")

    ner_pipe = load_transformer_ner()
    nlp = load_spacy()
    matcher, phrase_matcher = build_matchers(nlp)

    patient_raw = load_transcript("patient_text.txt")
    doctor_raw = load_transcript("doctor_text.txt")

    if not patient_raw and not doctor_raw:
        raise FileNotFoundError("Both patient_text.txt and doctor_text.txt are missing!")

    patient_ents = extract_for_speaker(patient_raw, ner_pipe, nlp, matcher, phrase_matcher, is_patient=True)
    doctor_ents = extract_for_speaker(doctor_raw, ner_pipe, nlp, matcher, phrase_matcher, is_patient=False)

    result = {"Patient": patient_ents, "Doctor": doctor_ents}
    out_path = os.path.join(DATA_DIR, "ner_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\nRESULTS")
    print("=" * 60)
    for speaker, data in result.items():
        print(f"\n{speaker.upper()} ENTITIES:")
        for cat, items in data.items():
            if items:
                print(f"  • {cat}:")
                for it in items:
                    print(f"      - {it}")
    print(f"\n[SUCCESS] Saved to {out_path}")
    return result


if __name__ == "__main__":
    run_ner_extraction()
