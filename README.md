# 🩺 Physician Notetaker – AI Medical Documentation System

An end-to-end AI pipeline that automatically converts doctor-patient conversations into structured **SOAP notes**, leveraging advanced NLP and transformer-based models for medical text understanding.

---

## 📘 Table of Contents

1. [Executive Summary](#executive-summary)  
2. [System Architecture](#system-architecture)  
3. [Component 1: Named Entity Recognition (NER)](#component-1-named-entity-recognition-ner)  
4. [Component 2: Medical Summary Generation](#component-2-medical-summary-generation)  
5. [Component 3: Sentiment & Intent Analysis](#component-3-sentiment--intent-analysis)  
6. [Component 4: SOAP Note Generation](#component-4-soap-note-generation)  
7. [Results & Performance Analysis](#results--performance-analysis)  
8. [Challenges & Limitations](#challenges--limitations)  
9. [Installation & Setup](#installation--setup)  
10. [How to Run the Project](#how-to-run-the-project)

---

## 🚀 Executive Summary

**Physician Notetaker** is an AI-powered medical documentation assistant designed to reduce the burden of manual note-taking in clinical practice.  
It processes **raw doctor–patient transcripts**, extracts key medical entities, analyzes patient sentiment and intent, and finally generates structured **SOAP notes** ready for use in EHR systems.

### 🧠 Highlights

- **Hybrid NER System** – 86% F1-score by combining transformer and rule-based methods  
- **Multi-Task BioBERT** – Joint sentiment and intent classification  
- **Auto-Generated SOAP Notes** – Structured documentation for clinical workflows  
- **Modular Design** – Each component is independent and reusable  
- **Thorough Evaluation** – Overfitting analysis and model comparison included  

### ⚙️ Tech Stack

| Component | Technology | Purpose |
|------------|-------------|----------|
| **NER Model** | BioBERT (`d4data/biomedical-ner-all`) | Medical entity extraction |
| **Secondary NER** | scispaCy (`en_core_sci_md`) | Rule-based pattern matching |
| **Sentiment & Intent** | Fine-tuned BioBERT | Emotion and intent classification |
| **Framework** | HuggingFace Transformers, PyTorch | Deep learning infrastructure |
| **NLP Tools** | spaCy 3.7+ | Pattern matching and text cleaning |

---

## 🧩 System Architecture



---

## 🧠 Component 1: Named Entity Recognition (NER)

Medical text is full of abbreviations, domain-specific terms, and conversational expressions.  
A **hybrid NER** approach captures this complexity using multiple layers:

| Challenge | Solution |
|------------|-----------|
| Domain-specific medical language | BioBERT pre-training on PubMed + PMC |
| Informal expressions | Rule-based pattern matching |
| Abbreviations | Contextual and regex extraction |
| Contextual ambiguity | Transformer attention mechanism |

### 🔍 Why BioBERT?

Traditional ML models (Naive Bayes, SVMs, TF-IDF) fail to capture semantics.  
**BioBERT**, trained on over 470K biomedical papers, understands medical context:
> “myocardial infarction” = “heart attack” = “MI”

**Performance Benchmarks:**
- BC5CDR: 88.6% F1  
- NCBI Disease: 89.4% F1  
- Real-world medical conversations: ~86% F1  

### 🧰 Rule-Based and Regex Layer

spaCy’s `Matcher` and regex patterns handle conversational phrases like:
- “feeling dizzy”, “having trouble sleeping”
- “diagnosed with whiplash injury”
- “10 sessions of physiotherapy”

This ensures comprehensive coverage of both clinical and everyday medical language.

---

## 🩹 Component 2: Medical Summary Generation

**Goal:** Convert raw NER output into a concise, clinically meaningful summary.

### Workflow:
1. **Entity Deduplication** – keep longest unique phrases.  
2. **Patient Identification** – extract names using regex.  
3. **Contextual Mapping** – determine if entities represent symptoms, diagnoses, or treatments.  

### Example Output
```json
{
  "Patient_Name": "Janet Jones",
  "Chief_Complaint": "Neck and back pain following motor vehicle accident",
  "Symptoms": ["Severe neck pain", "Back pain", "Sleep disturbance"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 sessions of physiotherapy", "Analgesics"],
  "Prognosis": "Full recovery expected within 6 months"
}


# Component 3: Named Entity Recognition (NER) Extractor
# ------------------------------------------------------

# Description:
# Extracts key medical entities such as Symptoms, Diagnosis, and Treatment
# from the doctor-patient transcript using pretrained NLP models.

# Run the NER extractor directly
python src/ner_extractor.py --input data/sample_transcript.txt

# Example Output:
# {
#     "Symptoms": ["shoulder pain", "numbness", "dizziness"],
#     "Diagnosis": ["shoulder injury"],
#     "Treatment": ["physiotherapy", "painkillers"]
# }

# Notes:
# - Uses scispaCy / BioClinicalBERT for medical entity recognition
# - Handles missing or ambiguous data automatically
# - Output is saved as JSON for use by other pipeline components


# Component 4: Medical Summarizer
# -------------------------------

# Description:
# Generates a concise and structured summary of the medical conversation.
# Integrates extracted entities from the NER step with contextual transcript data.

# Run the summarizer directly
python src/summarizer.py --transcript data/sample_transcript.txt --ner_output data/ner_results.json

# Example Output:
# "Patient David Thompson reports persistent shoulder pain, dizziness, and numbness.
# Diagnosis: Shoulder injury. Advised physiotherapy sessions and prescribed painkillers."

# Notes:
# - Uses transformer-based summarization models like Flan-T5 or BioBART
# - Handles incomplete or uncertain information contextually
# - Output is a human-readable summary used by the SOAP generator


# Component 5: Pipeline Runner
# ----------------------------

# Description:
# Orchestrates the entire pipeline — from transcript input to final SOAP note.
# Runs NER → Summarization → Sentiment → SOAP generation automatically.

# Run full pipeline
python src/pipeline_runner.py --input data/sample_transcript.txt

# Example Output:
#  NER Extraction Complete
#  Summary Generated
#  Sentiment & Intent Analyzed
#  SOAP Note Created: output/soap_note.json

# Notes:
# - Automates all steps sequentially
# - Ensures consistent and reproducible outputs
# - Ideal for deployment in clinical NLP systems



# Clone the repository
git clone https://github.com/yourusername/Physician-Notetaker.git
cd Physician-Notetaker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt

# Core Libraries
spacy==3.7.2
scispacy==0.5.2
transformers==4.45.0
torch==2.3.0
pandas==2.2.3
nltk==3.9
json
argparse


