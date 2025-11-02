import os
import re
from typing import Tuple

def load_transcript(path: str) -> str:
    """Load raw transcript text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()  # normalize case
    # remove common fillers and hesitations
    text = re.sub(r"\b(uh|umm|hmm|yeah|okay|right|you know|huh)\b", "", text)
    # remove any unwanted characters except punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", " ", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_dialogue(text: str) -> Tuple[list, list]:
    """Separate transcript into doctor and patient dialogues."""
    doctor_lines, patient_lines = [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("doctor:"):
            cleaned = clean_text(line.replace("Doctor:", "").strip())
            if cleaned:
                doctor_lines.append(cleaned)
        elif line.lower().startswith("patient:"):
            cleaned = clean_text(line.replace("Patient:", "").strip())
            if cleaned:
                patient_lines.append(cleaned)
    return doctor_lines, patient_lines

def save_processed(doctor_lines, patient_lines, out_dir="data/processed"):
    """Save processed doctor and patient text to separate files."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "doctor_text.txt"), "w", encoding="utf-8") as d:
        d.write("\n".join(doctor_lines))
    with open(os.path.join(out_dir, "patient_text.txt"), "w", encoding="utf-8") as p:
        p.write("\n".join(patient_lines))

if __name__ == "__main__":
    text = load_transcript("data/raw_transcript.txt")
    doctor, patient = parse_dialogue(text)
    save_processed(doctor, patient)
    print(" Processed and cleaned transcript saved in data/processed/")
