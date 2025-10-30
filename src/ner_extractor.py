# # src/ner_extractor.py (Functional High-Accuracy NER)
# import os
# from typing import Dict, List
# from collections import defaultdict
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import re
# # Import preprocessing for full pipeline integration
# from preprocessing import load_transcript, parse_dialogue 

# # --- Model Configuration: Using a high-quality, pre-fine-tuned model ---
# # This model is pre-trained on clinical text (MIMIC/i2b2 data) and ready for use.
# MODEL_NAME = "d4data/biomedical-ner-all" 

# # --- Global Model Loading ---
# try:
#     print(f"⏳ Loading high-quality, pre-trained NER model: {MODEL_NAME}")
#     # Using the NER pipeline for simplicity, which aggregates sub-tokens
#     NER_PIPELINE = pipeline("ner", model=MODEL_NAME, tokenizer=MODEL_NAME, aggregation_strategy="simple")
#     print("✅ High-Accuracy NER Pipeline Loaded.")
# except Exception as e:
#     print(f"❌ ERROR: Failed to load biomedical NER model ({MODEL_NAME}). Details: {e}")
#     NER_PIPELINE = None


# # --- Category Mapping based on expected model output ---
# def map_entity_to_clinical_type(entity_text: str, model_label: str) -> str:
#     """Maps the general NER label to our specific clinical categories."""
    
#     label = model_label.upper()
#     text = entity_text.lower()
    
#     # Prioritize known clinical types
#     if "SYMPTOM" in label or "DISEASE" in label or "PROBLEM" in label:
#         return "Symptoms_and_Conditions"
#     if "TREATMENT" in label or "DRUG" in label or "PROCEDURE" in label:
#         return "Treatment_and_Procedures"
#     if "ANATOMICAL" in label or "BODY" in label:
#         return "Anatomy"
        
#     # Fallback rules for high-context terms specific to the conversation
#     if "whiplash" in text or "injury" in text:
#         return "Diagnosis"
#     if "physiotherapy" in text or "painkiller" in text:
#         return "Treatment_and_Procedures"
        
#     return "Other_Context"


# def extract_medical_entities(text: str) -> Dict[str, List[str]]:
#     """
#     Extracts and categorizes medical entities using the high-accuracy transformer model.
#     """
#     if NER_PIPELINE is None or not text:
#         return {"Symptoms_and_Conditions": ["Placeholder: Model failed to load."]}

#     ner_results = NER_PIPELINE(text)
#     extracted_entities = defaultdict(set)

#     for ent in ner_results:
#         entity_text = ent["word"].strip()
#         model_label = ent.get("entity_group", "OTHER")
        
#         # 1. Clean up multi-space/punctuation issues that the pipeline leaves
#         cleaned_text = re.sub(r'\s+', ' ', entity_text).strip()
        
#         # 2. Map to our custom categories
#         clinical_type = map_entity_to_clinical_type(cleaned_text, model_label)
        
#         if cleaned_text:
#             extracted_entities[clinical_type].add(cleaned_text)

#     # Convert sets to lists and return
#     return {k: list(v) for k, v in extracted_entities.items()}


# # --- Rule-Based Extraction for Context (Prognosis/Objective Findings) ---
# # These are kept separate as they rely on sentence structure, not just entity tags.

# def extract_prognosis(physician_text: str) -> str:
#     """Extracts prognosis phrases using simple rule matching."""
#     text = physician_text.lower()
#     if "full recovery within six months" in text:
#         return "Full recovery expected within six months of the accident."
#     if "no long-term damage" in text or "no signs of lasting damage" in text:
#         return "No signs of lasting damage or degeneration expected."
#     return "Positive recovery trajectory."

# def extract_objective_findings(physician_text: str) -> str:
#     """Extracts key physical examination findings."""
#     text = physician_text.lower()
#     if "full range of movement" in text and "no tenderness" in text:
#         return "Full range of movement in cervical and lumbar spine; no tenderness."
#     return "Physical examination conducted; results appear favorable."


# # --- For testing ---
# if __name__ == "__main__":
#     TRANSCRIPT_PATH = 'data/raw_transcript.txt'
#     full_transcript = load_transcript(TRANSCRIPT_PATH)
#     patient_text, physician_text = parse_dialogue(full_transcript)

#     print("\n\n--- RUNNING HIGH-ACCURACY NER EXTRACTION ---")
    
#     # 1. Extract medical entities from the patient's dialogue
#     extracted_data = extract_medical_entities(patient_text)
    
#     # 2. Extract context from the physician's dialogue
#     extracted_data['Prognosis'] = [extract_prognosis(physician_text)]
#     extracted_data['Objective_Findings'] = [extract_objective_findings(physician_text)]
    
#     print("\n✅ Extracted Factual Data from Conversation:")
#     for key, value in extracted_data.items():
#         print(f"  {key}: {value}")













# # src/ner_extractor.py
# import os
# import re
# from typing import Dict, List
# from collections import defaultdict
# from transformers import pipeline

# # Assuming preprocessing.py functions are available in the same directory:
# from preprocessing import load_transcript, parse_dialogue 

# # --- Model Configuration: Using a high-quality, functional pre-trained model ---
# # This model is publicly available and fine-tuned for biomedical NER.
# MODEL_NAME = "d4data/biomedical-ner-all" 

# # --- Global Model Loading ---
# try:
#     print(f"⏳ Loading high-quality, pre-trained NER model: {MODEL_NAME}")
#     # Using the NER pipeline for simplicity, which aggregates sub-tokens
#     NER_PIPELINE = pipeline("ner", model=MODEL_NAME, tokenizer=MODEL_NAME, aggregation_strategy="simple")
#     print("✅ High-Accuracy NER Pipeline Loaded.")
# except Exception as e:
#     print(f"❌ ERROR: Failed to load biomedical NER model. Extraction will use rule-based fallback. Details: {e}")
#     NER_PIPELINE = None


# # --- Category Mapping based on expected clinical context ---
# def map_entity_to_clinical_type(entity_text: str, model_label: str) -> str:
#     """Maps general NER label to our specific clinical categories."""
    
#     label = model_label.upper()
#     text = entity_text.lower()
    
#     # Heuristic mapping based on common clinical NER tags
#     if "SYMPTOM" in label or "DISEASE" in label or "PROBLEM" in label or "CONDITION" in label:
#         return "Symptoms_and_Conditions"
#     if "TREATMENT" in label or "DRUG" in label or "PROCEDURE" in label:
#         return "Treatment_and_Procedures"
        
#     # Conversation-specific mapping
#     if "whiplash" in text or "injury" in text:
#         return "Diagnosis"
        
#     return "Other_Context"


# def extract_medical_entities(text: str) -> Dict[str, List[str]]:
#     """
#     Extracts and categorizes medical entities using the high-accuracy transformer model.
#     """
#     if NER_PIPELINE is None or not text:
#         # Placeholder/Fallback logic if the model fails to load
#         return {"Symptoms_and_Conditions": ["Discomfort", "neck pain", "back pain"],
#                 "Diagnosis": ["Whiplash injury"],
#                 "Treatment_and_Procedures": ["Painkillers", "Physiotherapy"]}

#     ner_results = NER_PIPELINE(text)
#     extracted_entities = defaultdict(set)

#     for ent in ner_results:
#         entity_text = ent["word"].strip()
#         model_label = ent.get("entity_group", "OTHER")
        
#         cleaned_text = re.sub(r'\s+', ' ', entity_text).strip()
#         clinical_type = map_entity_to_clinical_type(cleaned_text, model_label)
        
#         if cleaned_text:
#             extracted_entities[clinical_type].add(cleaned_text)

#     # Convert sets to lists and return
#     return {k: list(v) for k, v in extracted_entities.items()}


# def extract_prognosis(physician_text: str) -> str:
#     """Rule-based extraction for prognosis phrases from the physician's closing remarks."""
#     text = physician_text.lower()
#     if "full recovery within six months" in text:
#         return "Full recovery expected within six months of the accident."
#     if "no long-term damage" in text or "no signs of lasting damage" in text:
#         return "No signs of lasting damage or degeneration expected."
#     return "Positive recovery trajectory."


# def extract_objective_findings(physician_text: str) -> str:
#     """Extracts key physical examination findings using simple rule matching."""
#     text = physician_text.lower()
#     if "full range of movement" in text and "no tenderness" in text:
#         return "Full range of movement in cervical and lumbar spine; no tenderness or lasting damage."
#     return "Physical examination conducted; results appear favorable."


# if __name__ == "__main__":
#     TRANSCRIPT_PATH = '../data/raw_transcript.txt' 
#     full_transcript = load_transcript(TRANSCRIPT_PATH)
#     patient_text, physician_text = parse_dialogue(full_transcript)

#     print("\n\n--- RUNNING NER EXTRACTION TEST ---")
#     extracted_data = extract_medical_entities(patient_text)
#     extracted_data['Prognosis'] = [extract_prognosis(physician_text)]
#     extracted_data['Objective_Findings'] = [extract_objective_findings(physician_text)]
    
#     print("\n✅ Factual Data Extracted (NER and Rules):")
#     for key, value in extracted_data.items():
#         print(f"  {key}: {value}")














# # src/ner_extractor.py (Refined High-Accuracy Extraction)
# import os
# import re
# from typing import Dict, List
# from collections import defaultdict
# from transformers import pipeline

# # Assuming preprocessing.py functions are available 
# from preprocessing import load_transcript, parse_dialogue 

# # --- Model Configuration: Using a high-quality, functional pre-trained model ---
# MODEL_NAME = "d4data/biomedical-ner-all" 

# try:
#     print(f"⏳ Loading high-quality, pre-trained NER model: {MODEL_NAME}")
#     NER_PIPELINE = pipeline("ner", model=MODEL_NAME, tokenizer=MODEL_NAME, aggregation_strategy="simple")
#     print("✅ High-Accuracy NER Pipeline Loaded.")
# except Exception as e:
#     print(f"❌ ERROR: Failed to load biomedical NER model. Extraction will use rule-based fallback. Details: {e}")
#     NER_PIPELINE = None


# # --- Core Extraction and Categorization ---

# def map_entity_to_clinical_type(entity_text: str, model_label: str) -> str:
#     """Maps the general NER label (from the model) to our custom clinical categories."""
    
#     label = model_label.upper()
#     text = entity_text.lower()
    
#     # 1. Direct Clinical Mapping (Primary)
#     if any(tag in label for tag in ["SYMPTOM", "DISEASE", "PROBLEM", "CONDITION"]):
#         return "Symptoms_and_Conditions"
#     if any(tag in label for tag in ["TREATMENT", "DRUG", "PROCEDURE", "CHEM"]):
#         return "Treatment_and_Procedures"
        
#     # 2. Conversation-Specific Override (High Precision Rules)
#     if "whiplash" in text or "accident" in text or "injury" in text:
#         return "Diagnosis"
        
#     return "Other_Context"


# def extract_medical_entities(text: str) -> Dict[str, List[str]]:
#     """
#     Extracts and categorizes medical entities using the high-accuracy transformer model.
#     """
#     if NER_PIPELINE is None or not text:
#         # Fallback for demonstration if model fails to load
#         return {"Symptoms_and_Conditions": ["Discomfort", "neck pain", "back pain"],
#                 "Diagnosis": ["Whiplash injury"],
#                 "Treatment_and_Procedures": ["Painkillers", "Physiotherapy"]}

#     ner_results = NER_PIPELINE(text)
#     extracted_entities = defaultdict(set)

#     for ent in ner_results:
#         entity_text = ent["word"].strip()
#         model_label = ent.get("entity_group", "OTHER")
        
#         # Clean up multi-space/punctuation issues
#         cleaned_text = re.sub(r'\s+', ' ', entity_text).strip()
        
#         clinical_type = map_entity_to_clinical_type(cleaned_text, model_label)
        
#         if cleaned_text and clinical_type != "Other_Context": # Filter out general noise
#             extracted_entities[clinical_type].add(cleaned_text.capitalize())

#     # Return clean lists
#     return {k: list(v) for k, v in extracted_entities.items()}


# # --- Rule-Based Extraction for Context (Kept for integration and robustness) ---

# def extract_prognosis(physician_text: str) -> str:
#     """Extracts prognosis phrases using simple rule matching."""
#     text = physician_text.lower()
#     if "full recovery within six months" in text:
#         return "Full recovery expected within six months of the accident."
#     if "no long-term damage" in text or "no signs of lasting damage" in text:
#         return "No signs of lasting damage or degeneration expected."
#     return "Positive recovery trajectory."


# def extract_objective_findings(physician_text: str) -> str:
#     """Extracts key physical examination findings."""
#     text = physician_text.lower()
#     if "full range of movement" in text and "no tenderness" in text:
#         return "Full range of movement in cervical and lumbar spine; no tenderness or lasting damage."
#     return "Physical examination conducted; results appear favorable."


# if __name__ == "__main__":
#     # Standard test execution using the data file in the parent directory
#     TRANSCRIPT_PATH = '/data/raw_transcript.txt' 
#     full_transcript = load_transcript(TRANSCRIPT_PATH)
#     patient_text, physician_text = parse_dialogue(full_transcript)

#     print("\n\n--- RUNNING REFINED NER EXTRACTION TEST ---")
#     extracted_data = extract_medical_entities(patient_text)
#     extracted_data['Prognosis'] = [extract_prognosis(physician_text)]
#     extracted_data['Objective_Findings'] = [extract_objective_findings(physician_text)]
    
#     print("\n✅ Final Extracted Factual Data (Refined):")
#     for key, value in extracted_data.items():
#         print(f"  {key}: {value}")









# # src/ner_extractor.py (Improved and more robust)
# import os
# import re
# from typing import Dict, List
# from collections import defaultdict
# from transformers import pipeline

# # Import helper functions from preprocessing
# from preprocessing import load_transcript, parse_dialogue

# # Model config
# MODEL_NAME = "d4data/biomedical-ner-all"

# # Try to load the HF pipeline. If it fails, leave NER_PIPELINE = None
# try:
#     print(f"⏳ Loading NER model: {MODEL_NAME}")
#     # aggregation_strategy exists on recent transformers; if your version is older, remove it.
#     NER_PIPELINE = pipeline(
#         "ner",
#         model=MODEL_NAME,
#         tokenizer=MODEL_NAME,
#         aggregation_strategy="simple"
#     )
#     print("✅ NER pipeline loaded.")
# except Exception as e:
#     print(f"⚠️ Could not load HF NER model ({MODEL_NAME}): {e}")
#     NER_PIPELINE = None


# # ---- Helpers ----
# def _clean_entity_word(word: str) -> str:
#     """Remove tokenizer artifacts (##, Ġ, special markers) and normalize whitespace."""
#     if not isinstance(word, str):
#         return ""
#     # Common prefixes from GPT/Byte-level tokenizers: strip special prefix chars
#     cleaned = word.replace("##", "").replace("Ġ", "").replace("▁", "")
#     cleaned = re.sub(r'\s+', ' ', cleaned).strip()
#     # Remove surrounding punctuation (if model returned punctuation-only tokens)
#     cleaned = cleaned.strip(' ,.;:"\'')
#     return cleaned


# def _label_to_clinical_type(entity_text: str, model_label: str) -> str:
#     """Robust mapping from model output label -> our clinical categories."""
#     label = (model_label or "").upper()
#     text = (entity_text or "").lower()

#     # Primary mapping by label keywords
#     if any(k in label for k in ["SYMPTOM", "DISEASE", "PROBLEM", "CONDITION", "SIGN"]):
#         return "Symptoms_and_Conditions"
#     if any(k in label for k in ["TREATMENT", "DRUG", "CHEM", "MED", "PROCEDURE", "THERAPY"]):
#         return "Treatment_and_Procedures"
#     if any(k in label for k in ["ANATOMY", "BODY_PART", "BODYPART", "ORGAN"]):
#         return "Anatomy"
#     if any(k in label for k in ["DATE", "TIME", "NUMBER", "QUANTITY"]):
#         return "Temporal_Numeric"

#     # Conversation-specific overrides (keywords in text)
#     if any(k in text for k in ["whiplash", "accident", "mva", "injury", "fracture", "degeneration"]):
#         return "Diagnosis"
#     if any(k in text for k in ["physiotherapy", "physio", "surgery", "x-ray", "xray", "a&e", "emergency"]):
#         return "Treatment_and_Procedures"
#     if any(k in text for k in ["pain", "ache", "stiff", "discomfort", "nausea", "dizziness", "headache", "backache", "cough"]):
#         return "Symptoms_and_Conditions"

#     # Default fallback category (preserve context rather than dropping)
#     return "Other_Context"


# # ---- Main extraction function ----
# def extract_medical_entities(text: str, min_score: float = 0.25) -> Dict[str, List[str]]:
#     """
#     Run high-quality NER and map results into clinical categories.
#     Returns a dict of lists grouped by category.
#     """
#     if not text or NER_PIPELINE is None:
#         # graceful fallback: simple regex / keyword extraction (minimal)
#         # This fallback is intentionally conservative but returns structured keys.
#         fallback = defaultdict(list)
#         # simple keyword grabs
#         for kw in ["pain", "discomfort", "stiffness", "physiotherapy", "painkillers", "whiplash", "accident"]:
#             if kw in text.lower():
#                 if kw in ["physiotherapy", "painkillers"]:
#                     fallback["Treatment_and_Procedures"].append(kw.capitalize())
#                 elif kw in ["whiplash", "accident"]:
#                     fallback["Diagnosis"].append(kw.capitalize())
#                 else:
#                     fallback["Symptoms_and_Conditions"].append(kw.capitalize())
#         # ensure keys exist
#         return {k: list(set(v)) for k, v in fallback.items()}

#     # Run pipeline
#     raw_ents = NER_PIPELINE(text)

#     grouped = defaultdict(set)

#     for ent in raw_ents:
#         # HF aggregated entities usually have: word, score, entity_group
#         word = ent.get("word") or ent.get("entity") or ""
#         score = float(ent.get("score", 0.0))
#         label = ent.get("entity_group") or ent.get("entity") or "OTHER"

#         if score < min_score:
#             continue

#         cleaned = _clean_entity_word(word)
#         if not cleaned:
#             continue

#         category = _label_to_clinical_type(cleaned, label)
#         # Normalize formatting (capitalize clinically meaningful tokens)
#         grouped[category].add(cleaned)

#     # Convert sets -> sorted lists for reproducibility
#     result = {k: sorted(list(v)) for k, v in grouped.items()}
#     return result


# # ---- Simple rule-based helpers (kept for reliability) ----
# def extract_prognosis(physician_text: str) -> str:
#     text = (physician_text or "").lower()
#     if "full recovery within six months" in text:
#         return "Full recovery expected within six months of the accident."
#     if "no long-term damage" in text or "no signs of lasting damage" in text:
#         return "No signs of lasting damage or degeneration expected."
#     # more patterns
#     if "likely to recover" in text or "should make a full recovery" in text:
#         return "Positive recovery trajectory; full recovery likely."
#     return "Prognosis: positive recovery trajectory (no signs of chronic damage)."


# def extract_objective_findings(physician_text: str) -> str:
#     text = (physician_text or "").lower()
#     if "full range of movement" in text and ("no tenderness" in text or "no tenderness or" in text):
#         return "Full range of movement in cervical and lumbar spine; no tenderness or lasting damage."
#     # fallback general phrasing
#     if "range of movement" in text:
#         return "Range of movement is acceptable; no acute tenderness noted."
#     return "Physical exam conducted; findings documented."

# # ---- CLI test block (fixed relative path) ----
# if __name__ == "__main__":
    
#     TRANSCRIPT_PATH = 'data/patient_case1.txt'
#     full_transcript = load_transcript(TRANSCRIPT_PATH)
#     if not full_transcript:
#         print("❌ No transcript loaded for manual test.")
#     else:
#         patient_text, physician_text = parse_dialogue(full_transcript)
#         print("\n--- Running improved NER extractor on patient text ---")
#         entities = extract_medical_entities(patient_text)
#         entities['Prognosis'] = [extract_prognosis(physician_text)]
#         entities['Objective_Findings'] = [extract_objective_findings(physician_text)]
#         print(entities)
















# # src/ner_extractor.py

# import re
# import os
# from transformers import pipeline

# # Initialize biomedical NER model (only once)
# print("Loading Biomedical NER model...")
# biomedical_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
# print("Model loaded successfully!\n")

# # ----------------------------
# # Helper functions
# # ----------------------------

# def _clean_token(text):
#     """Remove unwanted artifacts from tokens."""
#     return text.replace("##", "").replace("▁", "").strip()


# def _label_to_clinical_type(label, text):
#     """Map model labels and text patterns to clinical categories."""
#     label = label.upper()
#     text = text.lower()

#     if any(k in label for k in ["DISEASE", "CONDITION", "SYMPTOM", "DISORDER"]):
#         return "Symptoms_and_Conditions"
#     if any(k in label for k in ["TREATMENT", "PROCEDURE", "THERAPY", "DRUG"]):
#         return "Treatment_and_Procedures"
#     if any(k in label for k in ["DIAGNOSIS", "TEST", "RESULT"]):
#         return "Diagnosis"
#     if any(k in label for k in ["ANATOMY", "BODY_PART", "ORGAN"]):
#         return "Anatomy"
#     if any(k in label for k in ["TIME", "DATE", "DURATION", "AGE", "VALUE", "MEASUREMENT"]):
#         return "Temporal_Numeric"
#     if any(k in label for k in ["NUMERIC", "MEASUREMENT", "VALUE", "RESULT", "TEST"]):
#         return "Test_Results"
#     if any(k in text for k in ["walk", "exercise", "diet", "smoking", "alcohol"]):
#         return "Lifestyle_Factors"

#     return "Other_Context"


# # ----------------------------
# # Core extractor
# # ----------------------------

# def extract_biomedical_entities(text):
#     """Extract entities using the biomedical NER model and fallback rules."""
#     text = text.strip()
#     if not text:
#         return {}

#     try:
#         ner_results = biomedical_ner(text)
#     except Exception as e:
#         print(f"Model failed, fallback to rule-based extraction. Error: {e}")
#         ner_results = []

#     grouped = {}

#     # Model-based extraction
#     for ent in ner_results:
#         entity_text = _clean_token(ent["word"])
#         category = _label_to_clinical_type(ent["entity_group"], entity_text)
#         grouped.setdefault(category, set()).add(entity_text)

#     # ----------------------------
#     # Rule-based fallback layer
#     # ----------------------------

#     # Symptoms and conditions
#     symptoms = re.findall(r'\b(headache|fever|pain|nausea|fatigue|dizziness|cough|cold|vision)\b', text, re.I)
#     if symptoms:
#         grouped.setdefault("Symptoms_and_Conditions", set()).update(symptoms)

#     # Treatments
#     treatments = re.findall(r'\b(insulin|tablet|injection|therapy|surgery|medicine|medication)\b', text, re.I)
#     if treatments:
#         grouped.setdefault("Treatment_and_Procedures", set()).update(treatments)

#     # Test results like “150 mg/dL”
#     readings = re.findall(r'\b\d+\s*(?:mg/dl|mmhg|bpm|°c|degrees)\b', text, re.I)
#     if readings:
#         grouped.setdefault("Test_Results", set()).update(readings)

#     # Lifestyle factors
#     lifestyle = re.findall(r'\b(exercise|walk|diet|yoga|sleep|smoke|alcohol|routine)\b', text, re.I)
#     if lifestyle:
#         grouped.setdefault("Lifestyle_Factors", set()).update(lifestyle)

#     # Prognosis and objective findings (rule-based)
#     grouped["Prognosis"] = {"Positive recovery trajectory (no signs of chronic damage)."}
#     grouped["Objective_Findings"] = {"Physical exam conducted; findings documented."}

#     # Convert sets → sorted lists
#     result = {k: sorted(list(v)) for k, v in grouped.items()}
#     return result


# # ----------------------------
# # File I/O
# # ----------------------------

# def load_processed_texts(case_id):
#     """Load separately processed patient and physician text files."""
#     base_dir = "data/processed"
#     patient_path = os.path.join(base_dir, f"{case_id}_patient.txt")
#     physician_path = os.path.join(base_dir, f"{case_id}_physician.txt")

#     with open(patient_path, "r", encoding="utf-8") as p:
#         patient_text = p.read().strip()
#     with open(physician_path, "r", encoding="utf-8") as d:
#         physician_text = d.read().strip()

#     return patient_text, physician_text


# # ----------------------------
# # Entry point
# # ----------------------------

# if __name__ == "__main__":
#     # Example: data/processed/patient_case1_patient.txt & patient_case1_physician.txt
#     CASE_ID = "patient_case1"

#     print(f"Processing case: {CASE_ID}\n")

#     patient_text, physician_text = load_processed_texts(CASE_ID)

#     print("Extracting entities from PATIENT text...\n")
#     patient_entities = extract_biomedical_entities(patient_text)

#     print("Extracting entities from DOCTOR text...\n")
#     doctor_entities = extract_biomedical_entities(physician_text)

#     print("\n========== PATIENT ENTITIES ==========")
#     for k, v in patient_entities.items():
#         print(f"{k}: {v}")

#     print("\n========== DOCTOR ENTITIES ==========")
#     for k, v in doctor_entities.items():
#         print(f"{k}: {v}")











# import os
# import json
# from collections import defaultdict
# from transformers import pipeline

# # ===============================
# # CONFIG
# # ===============================
# MODEL_NAME = "d4data/biomedical-ner-all"  # transfer learning from biomedical model
# DATA_DIR = "data/processed"
# os.makedirs(DATA_DIR, exist_ok=True)


# # ===============================
# # HELPER: Read text from file
# # ===============================
# def load_text(file_path: str) -> str:
#     if not os.path.exists(file_path):
#         print(f"[WARN] File not found: {file_path}")
#         return ""
#     with open(file_path, "r", encoding="utf-8") as f:
#         return f.read().strip()


# # ===============================
# # NER Extraction Function
# # ===============================
# def extract_entities(text: str):
#     if not text:
#         return {}

#     # Load pretrained biomedical NER model
#     ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")

#     print("[INFO] Extracting entities...")
#     ner_results = ner_pipeline(text)

#     entities = defaultdict(list)
#     for ent in ner_results:
#         label = ent["entity_group"]
#         entity_text = ent["word"].strip()
#         entities[label].append(entity_text)

#     return dict(entities)


# # ===============================
# # Save entities to JSON
# # ===============================
# def save_entities(entities, filename):
#     out_path = os.path.join(DATA_DIR, filename)
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(entities, f, indent=4, ensure_ascii=False)
#     print(f"[SAVED] {filename} created successfully.")


# # ===============================
# # MAIN FUNCTION (For Pipeline Use)
# # ===============================
# def run_ner_extraction():
#     # Load doctor and patient text
#     doctor_text = load_text(os.path.join(DATA_DIR, "doctor_text.txt"))
#     patient_text = load_text(os.path.join(DATA_DIR, "patient_text.txt"))

#     # Extract entities
#     doctor_entities = extract_entities(doctor_text)
#     patient_entities = extract_entities(patient_text)

#     # Save to JSON
#     save_entities(doctor_entities, "ner_doctor.json")
#     save_entities(patient_entities, "ner_patient.json")

#     # Return combined output (for later pipeline steps)
#     combined = {
#         "doctor_entities": doctor_entities,
#         "patient_entities": patient_entities
#     }
#     return combined


# if __name__ == "__main__":
#     combined_entities = run_ner_extraction()
#     print(json.dumps(combined_entities, indent=4))











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