# main.py
import os
import json
from ner_to_summary import main as ner_summary_main
from multitask_finetune import run_pipeline as sentiment_pipeline
from soap_generator_auto import main as soap_main

# Directories
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Generate Summary (ner_to_summary.py)
# ----------------------------------------------------------------------
print("Running NER → Summary module...")
summary = ner_summary_main()  # This will generate summary.json in processed
# Save a copy in output folder
processed_summary_file = os.path.join(PROCESSED_DIR, "summary.json")
output_summary_file = os.path.join(OUTPUT_DIR, "summary.json")
if os.path.exists(processed_summary_file):
    with open(processed_summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    with open(output_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"[SUCCESS] Summary copied to {output_summary_file}")
else:
    print("[WARNING] Processed summary.json not found!")

# ----------------------------------------------------------------------
# 2. Run Sentiment & Intent (sentiment_intent.py)
# ----------------------------------------------------------------------
print("\n Running Sentiment & Intent module...")
sentiment_output_file = os.path.join(OUTPUT_DIR, "sentiment_intent_results.json")
sentiment_pipeline(
    input_path=os.path.join(PROCESSED_DIR, "patient_text.txt"),
    output_path=sentiment_output_file
)
with open(sentiment_output_file, "r", encoding="utf-8") as f:
    sentiment_data = json.load(f)

# ----------------------------------------------------------------------
# 3. Run SOAP generator (soap_generator_auto.py)
# ----------------------------------------------------------------------
print("\n Running SOAP generator module...")
# Override the output path to save in OUTPUT_DIR
from soap_generator_auto import load_data, map_to_soap_auto, refine_text_with_llm

# Load NER, Summary, and Sentiment results
ner_file = os.path.join(PROCESSED_DIR, "ner_results.json")
summary_file = os.path.join(PROCESSED_DIR, "summary.json")

with open(ner_file, "r", encoding="utf-8") as f:
    ner_data = json.load(f)
with open(summary_file, "r", encoding="utf-8") as f:
    summary_data = json.load(f)
# sentiment_data already loaded

# Map to SOAP and refine
soap_note = map_to_soap_auto(ner_data, summary_data, sentiment_data)
refined_soap = refine_text_with_llm(soap_note)

# Save final SOAP note in output folder
soap_output_file = os.path.join(OUTPUT_DIR, "soap_note.json")
with open(soap_output_file, "w", encoding="utf-8") as f:
    json.dump(refined_soap, f, indent=4, ensure_ascii=False)
print(f"[SUCCESS] SOAP note saved at {soap_output_file}")

# ----------------------------------------------------------------------
# 4. Copy NER results to output folder
# ----------------------------------------------------------------------
ner_output_file = os.path.join(OUTPUT_DIR, "ner_results.json")
with open(ner_file, "r", encoding="utf-8") as f:
    ner_data = json.load(f)
with open(ner_output_file, "w", encoding="utf-8") as f:
    json.dump(ner_data, f, indent=4, ensure_ascii=False)
print(f"[SUCCESS] NER results copied to {ner_output_file}")

print("\n Pipeline complete! All outputs saved in data/output/")
