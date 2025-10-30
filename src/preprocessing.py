# # src/preprocessing.py
# import re
# from typing import List, Tuple
# import os

# def load_transcript(file_path: str) -> str:
#     """Loads the raw transcript text."""
    
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return f.read()
#     except FileNotFoundError:
#         print(f"Error: Transcript file not found at {file_path}")
#         return ""

# def parse_dialogue(transcript: str) -> Tuple[str, str]:
#     """
#     Separates the transcript into physician and patient text.
#     Returns: (patient_dialogue, physician_dialogue)
#     """
#     patient_lines = []
#     physician_lines = []
    
#     # Simple regex to split based on 'Speaker: ' pattern
#     lines = transcript.strip().split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if line.startswith("Patient:"):
#             # Clean the speaker tag and excess spacing
#             text = re.sub(r'^Patient:\s*', '', line).strip()
#             if text:
#                 patient_lines.append(text)
#         elif line.startswith("Physician:"):
#             text = re.sub(r'^Physician:\s*', '', line).strip()
#             if text:
#                 physician_lines.append(text)
#         # Handle the [Physical Examination Conducted] tag
#         elif not line.startswith("["):
#             # Include non-tagged lines if necessary, but for now, ignore them
#             pass

#     # Join lines back into continuous text for NLP processing
#     patient_dialogue = ' '.join(patient_lines)
#     physician_dialogue = ' '.join(physician_lines)
    
#     return patient_dialogue, physician_dialogue

# if __name__ == '__main__':
#     transcript_path = 'data/raw_transcript.txt'
#     full_transcript = load_transcript(transcript_path)
    
#     if full_transcript:
#         patient_text, physician_text = parse_dialogue(full_transcript)
        
#         print("--- Full Patient Dialogue ---")
#         print(patient_text[:200] + '...')
        
#         print("\n--- Full Physician Dialogue ---")
#         print(physician_text[:200] + '...')








# # src/preprocessing.py (FINAL CORRECTED VERSION)
# import re
# from typing import Tuple
# import os

# def load_transcript(file_path: str) -> str:
#     """
#     Loads the raw transcript text by intelligently resolving the file path 
#     relative to the project structure (assuming data/ is in the root).
#     """
    
#     # Calculate the project root directory (up two levels from src/preprocessing.py)
#     # This is safe when running main scripts from the project root.
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     full_path = os.path.join(base_dir, file_path)
    
#     try:
#         # Attempt to open using the absolute path derived from the project structure
#         with open(full_path, 'r', encoding='utf-8') as f:
#             return f.read()
#     except FileNotFoundError:
#         # Fallback for when the script is run directly from the src/ directory
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                  return f.read()
#         except FileNotFoundError:
#             print(f"Error: Transcript file not found at {file_path} or {full_path}")
#             return ""

# def parse_dialogue(transcript: str) -> Tuple[str, str]:
#     """
#     Separates the transcript into physician and patient text.
#     Returns: (patient_dialogue, physician_dialogue)
#     """
#     patient_lines = []
#     physician_lines = []
    
#     lines = transcript.strip().split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if line.startswith("Patient:"):
#             # Clean the speaker tag and excess spacing
#             text = re.sub(r'^Patient:\s*', '', line).strip()
#             if text:
#                 patient_lines.append(text)
#         elif line.startswith("Physician:"):
#             text = re.sub(r'^Physician:\s*', '', line).strip()
#             if text:
#                 physician_lines.append(text)
#         # We explicitly handle the [Physical Examination Conducted] tag by ignoring it
#         # and ignore any other non-tagged lines, focusing purely on dialogue.
#         elif line.startswith("["):
#              continue

#     # Join lines back into continuous text for NLP processing
#     patient_dialogue = ' '.join(patient_lines)
#     physician_dialogue = ' '.join(physician_lines)
    
#     return patient_dialogue, physician_dialogue

# if __name__ == '__main__':
#     # Test execution path assumes data/raw_transcript.txt is available
#     transcript_path = 'data/raw_transcript.txt'
#     full_transcript = load_transcript(transcript_path)
    
#     if full_transcript:
#         patient_text, physician_text = parse_dialogue(full_transcript)
        
#         print("--- Full Transcript Loaded and Preprocessed Successfully ---")
#         print(f"Patient Dialogue Start: {patient_text[:100]}...")
#         print(f"Physician Dialogue Start: {physician_text[:100]}...")











# # src/preprocessing.py
# import re
# import os
# from typing import Tuple

# def load_transcript(file_path: str) -> str:
#     """
#     Loads the raw transcript text by intelligently resolving the file path 
#     relative to the project structure (assuming data/ is in the root).
#     """
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     full_path = os.path.join(base_dir, file_path)
    
#     try:
#         with open(full_path, 'r', encoding='utf-8') as f:
#             return f.read()
#     except FileNotFoundError:
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 return f.read()
#         except FileNotFoundError:
#             print(f"❌ Error: Transcript file not found at {file_path} or {full_path}")
#             return ""


# def parse_dialogue(transcript: str) -> Tuple[str, str]:
#     """
#     Separates the transcript into physician and patient text.
#     Returns: (patient_dialogue, physician_dialogue)
#     """
#     patient_lines = []
#     physician_lines = []

#     lines = transcript.strip().split('\n')

#     for line in lines:
#         line = line.strip()
#         if line.startswith("Patient:"):
#             text = re.sub(r'^Patient:\s*', '', line).strip()
#             if text:
#                 patient_lines.append(text)
#         elif line.startswith("Doctor:") or line.startswith("Physician:"):
#             text = re.sub(r'^(Doctor|Physician):\s*', '', line).strip()
#             if text:
#                 physician_lines.append(text)
#         elif line.startswith("["):
#             continue

#     patient_dialogue = ' '.join(patient_lines)
#     physician_dialogue = ' '.join(physician_lines)

#     return patient_dialogue, physician_dialogue


# def save_preprocessed(patient_text: str, physician_text: str, input_path: str) -> str:
#     """
#     Saves preprocessed text files under data/processed/ with matching base names.
#     Returns path to the saved folder.
#     """
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     processed_dir = os.path.join(base_dir, "data", "processed")
#     os.makedirs(processed_dir, exist_ok=True)

#     base_name = os.path.splitext(os.path.basename(input_path))[0]

#     patient_out = os.path.join(processed_dir, f"{base_name}_patient.txt")
#     physician_out = os.path.join(processed_dir, f"{base_name}_physician.txt")

#     with open(patient_out, "w", encoding="utf-8") as f:
#         f.write(patient_text)
#     with open(physician_out, "w", encoding="utf-8") as f:
#         f.write(physician_text)

#     print(f"✅ Preprocessed data saved in: {processed_dir}")
#     return processed_dir


# if __name__ == '__main__':
#     # Example: python src/preprocessing.py data/patient_case1.txt
#     transcript_path = 'data/patient_case1.txt'
#     transcript = load_transcript(transcript_path)
    
#     if transcript:
#         patient_text, physician_text = parse_dialogue(transcript)
#         save_preprocessed(patient_text, physician_text, transcript_path)

#         print("--- ✅ Full Transcript Loaded and Preprocessed Successfully ---")
#         print(f"🧍 Patient Start: {patient_text[:100]}...")
#         print(f"👨‍⚕️ Physician Start: {physician_text[:100]}...")












import os
from typing import Tuple

def load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def parse_dialogue(text: str) -> Tuple[list, list]:
    doctor_lines, patient_lines = [], []
    for line in text.split("\n"):
        if line.lower().startswith("doctor:"):
            doctor_lines.append(line.replace("Doctor:", "").strip())
        elif line.lower().startswith("patient:"):
            patient_lines.append(line.replace("Patient:", "").strip())
    return doctor_lines, patient_lines

def save_processed(doctor_lines, patient_lines, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "doctor_text.txt"), "w", encoding="utf-8") as d:
        d.write("\n".join(doctor_lines))
    with open(os.path.join(out_dir, "patient_text.txt"), "w", encoding="utf-8") as p:
        p.write("\n".join(patient_lines))

if __name__ == "__main__":
    text = load_transcript("data/raw_transcript.txt")
    doctor, patient = parse_dialogue(text)
    save_processed(doctor, patient)
    print("✅ Processed transcript saved in data/processed/")
