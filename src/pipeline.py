# # src/pipeline.py
# import json
# import os
# from datetime import datetime
# from preprocessing import load_transcript, parse_dialogue
# from ner_extractor import extract_medical_entities, extract_prognosis, extract_objective_findings
# from sentiment_analyzer import analyze_sentiment_and_intent
# from summarizer import create_structured_summary, create_soap_note

# # Relative path to input transcript
# TRANSCRIPT_PATH = 'data/patient_case1.txt'
# OUTPUT_DIR = 'output'  # Folder where JSON results will be saved


# def run_physician_notetaker_pipeline():
#     print("=" * 80)
#     print("🩺 Physician Notetaker Pipeline: Starting Final Analysis 🩺")
#     print("=" * 80)

#     # --- Step 1: Load and preprocess data ---
#     data_path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', TRANSCRIPT_PATH)
#     full_transcript = load_transcript(data_path_root)
#     if not full_transcript:
#         print("❌ Transcript not found or empty.")
#         return

#     patient_text, physician_text = parse_dialogue(full_transcript)
#     print(f"✅ Data Preprocessed.")

#     # --- Step 2: Sentiment + Intent Analysis ---
#     sentiment_results = analyze_sentiment_and_intent(patient_text)
#     print(f"✅ Sentiment/Intent Analysis Complete.")

#     # --- Step 3: NER Extraction ---
#     ner_data = extract_medical_entities(patient_text)
#     print(f"✅ NER Extraction Complete.")

#     # --- Step 4: Structured Summaries ---
#     structured_summary = create_structured_summary(patient_text, physician_text)
#     soap_note = create_soap_note(patient_text, physician_text, sentiment_results)

#     # --- Step 5: Save Outputs ---
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = os.path.join(OUTPUT_DIR, f"physician_notetaker_output_{timestamp}.json")

#     final_output = {
#         "structured_summary": structured_summary,
#         "soap_note": soap_note,
#         "sentiment_results": sentiment_results,
#         "ner_data": ner_data
#     }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)

#     print(f"\n💾 Results saved to: {output_path}")

#     # --- Step 6: Display Output Summary ---
#     print("\n" + "=" * 40)
#     print("  FINAL DELIVERABLES ")
#     print("=" * 40)
#     print("\n--- STRUCTURED SUMMARY ---")
#     print(json.dumps(structured_summary, indent=2))

#     print("\n--- SOAP NOTE ---")
#     print(json.dumps(soap_note, indent=2))

#     print("\n" + "=" * 80)


# if __name__ == '__main__':
#     run_physician_notetaker_pipeline()












# src/pipeline_runner.py
import os
import json
import traceback

# Import the functions you actually implemented
from ner_extractor import run_ner_extraction
from sentiment_intent_analyzer import run_sentiment_intent
from summarizer import run_summarizer
from soap_generator import generate_soap

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_text(output_path: str, text: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    print("🚀 Starting Physician-Notetaker pipeline...\n")

    try:
        # 1) NER extraction (saves ner_doctor.json, ner_patient.json)
        print("1) Running NER extraction...")
        ner_result = run_ner_extraction()
        print("   → NER extraction completed.\n")

        # 2) Sentiment & Intent analysis (saves sentiment_intent.json)
        print("2) Running Sentiment & Intent analysis...")
        senti_result = run_sentiment_intent()
        print("   → Sentiment & Intent analysis completed.\n")

        # 3) Summarization (reads ner + sentiment files, saves combined_summary.json)
        print("3) Running Summarizer...")
        summary_result = run_summarizer()
        print("   → Summarization completed.\n")

        # 4) SOAP generation (reads combined_summary.json, saves soap_pairs.json)
        print("4) Generating SOAP note...")
        soap_result = generate_soap()
        print("   → SOAP note generation completed.\n")

               # 5) Save a human-readable SOAP text file (also keep JSON saved by soap_generator)
        soap_text = []

        def format_section(section_dict):
            if isinstance(section_dict, dict):
                return "\n".join([f"{k}: {v}" for k, v in section_dict.items()])
            return str(section_dict)

        soap_text.append("SUBJECTIVE:\n" + format_section(soap_result.get("Subjective", {})) + "\n")
        soap_text.append("OBJECTIVE:\n" + format_section(soap_result.get("Objective", {})) + "\n")
        soap_text.append("ASSESSMENT:\n" + format_section(soap_result.get("Assessment", {})) + "\n")
        soap_text.append("PLAN:\n" + format_section(soap_result.get("Plan", {})) + "\n")

        soap_text_full = "\n\n".join(soap_text)

        out_txt_path = os.path.join(OUTPUT_DIR, "soap_note.txt")
        save_text(out_txt_path, soap_text_full)
        print(f"✅ Final SOAP note saved (text): {out_txt_path}")


        # Also write a summary run-report
        report = {
            "ner_result_keys": list(ner_result.keys()) if isinstance(ner_result, dict) else None,
            "sentiment_intent": senti_result,
            "summary_present": bool(summary_result.get("summary_text")) if isinstance(summary_result, dict) else None,
            "soap_saved_to": os.path.join(DATA_PROCESSED, "soap_pairs.json")
        }
        with open(os.path.join(OUTPUT_DIR, "pipeline_run_report.json"), "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2, ensure_ascii=False)
        print(f"ℹ️  Pipeline run report saved: {os.path.join(OUTPUT_DIR, 'pipeline_run_report.json')}")

        print("\n🎉 Pipeline finished successfully.")

    except Exception as e:
        print("\n⛔ Pipeline failed with an error:")
        traceback.print_exc()
        print("\nMake sure:")
        print(" - You ran the command from the project root: `python src/pipeline_runner.py`")
        print(" - data/processed/doctor_text.txt and patient_text.txt exist")
        print(" - All required packages from requirements.txt are installed")
        raise e


if __name__ == "__main__":
    main()
