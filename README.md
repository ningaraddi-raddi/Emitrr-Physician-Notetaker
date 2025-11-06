# 🩺 Physician Notetaker – AI Medical Documentation System

An end-to-end AI pipeline that automatically converts doctor-patient conversations into structured **SOAP notes** using NLP and transformer-based models.

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [System Architecture](#system-architecture)  
3. [Key Components](#key-components)  
4. [Tech Stack](#tech-stack)  
5. [Why Pretrained Models Are Used](#why-pretrained-models-are-used)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [Project Structure](#project-structure)  
9. [Limitations](#limitations)  
10. [Future Work](#future-work)

---

## 🩺 Overview

**Physician Notetaker** automates medical documentation by processing raw doctor-patient transcripts and generating structured SOAP notes.  
The system uses a hybrid approach combining **BioBERT transformers** with **rule-based NLP methods**.

### Key Features

- 🧠 **Hybrid NER System** – Combines BioBERT + scispaCy for robust entity extraction  
- 💬 **Multi-Task Learning** – Joint sentiment and intent classification  
- 📋 **Auto SOAP Notes** – Structured clinical documentation  
- ⚙️ **Modular Pipeline** – Independent, reusable components  

---

## 🧩 System Architecture

```text
Doctor-Patient Transcript
         ↓
┌────────────────────────┐
│  1. NER Extraction     │  → Extract symptoms, diagnosis, treatment
└────────────────────────┘
         ↓
┌────────────────────────┐
│  2. Summarization      │  → Generate structured medical summary
└────────────────────────┘
         ↓
┌────────────────────────┐
│  3. Sentiment/Intent   │  → Analyze patient emotion & purpose
└────────────────────────┘
         ↓
┌────────────────────────┐
│  4. SOAP Generation    │  → Build clinical note (S.O.A.P format)
└────────────────────────┘
         ↓
    SOAP Note Output
🧠 Key Components
1. Named Entity Recognition (NER)
BioBERT Transformer – Pre-trained on 470K+ medical papers

scispaCy – Rule-based pattern matching with 100K+ medical terms

Regex Patterns – Extract measurements, dates, and quantities

Output: Symptoms, Diagnosis, Treatment, Prognosis

2. Medical Summary Generation
Deduplicates and cleans extracted entities

Extracts patient information using regex

Maps entities to medical categories

Output: Structured JSON summary

3. Sentiment & Intent Analysis
Multi-task BioBERT classifier with two heads:

Sentiment: Anxious / Neutral / Reassured

Intent: Reporting / Seeking / Expressing / Confirming

Fine-tuned on domain-relevant conversation samples

Output: Patient emotional state and conversation intent

4. SOAP Note Generation
S (Subjective): Patient complaints & history

O (Objective): Doctor observations & exam findings

A (Assessment): Diagnosis & clinical impression

P (Plan): Treatment & follow-up recommendations

Output: Clinical SOAP note in JSON format

⚙️ Tech Stack
Component	Technology	Purpose
NER	BioBERT (d4data/biomedical-ner-all)	Medical entity extraction
Secondary NER	scispaCy (en_core_sci_md)	Rule-based patterns
Sentiment/Intent	Fine-tuned BioBERT	Emotion & intent classification
Framework	HuggingFace Transformers, PyTorch	Deep learning
NLP Tools	spaCy 3.7+	Text processing

🧠 Why Pretrained Models Are Used
We adopted pretrained biomedical transformers like BioBERT and scispaCy because medical NLP tasks require understanding highly domain-specific language and terminology that are not captured by general NLP models.
Training from scratch would demand millions of medical records and heavy computation.

By using pretrained models, we leverage transfer learning, where models trained on vast biomedical literature already understand the context of clinical language.
Fine-tuning them on smaller, task-specific datasets provides strong performance even with limited labeled data.

Key Reasons
🩸 Domain-Specific Vocabulary: Medical dialogues include terms and abbreviations like HbA1c, BP, dyspnea, metformin, etc., which pretrained biomedical models already understand.

💰 Reduced Training Cost: Avoids training from scratch, saving time and compute resources.

🧩 Improved Contextual Understanding: BioBERT captures relationships between symptoms, diagnoses, and treatments with contextual accuracy.

🔁 Transfer Learning Advantage: Pretrained biomedical models generalize better on unseen medical conversations.

Summary
Task	Model Used	Purpose
NER Extraction	BioBERT (d4data/biomedical-ner-all)	Extracts entities like Symptoms, Diagnosis, and Treatment
Supplementary NER	scispaCy (en_core_sci_md)	Enhances coverage through rule-based medical term matching
Sentiment & Intent Analysis	Fine-tuned BioBERT	Detects patient emotions and conversational intent
SOAP Generation	Rule-based Template + Transformer Output	Creates structured clinical documentation

Note: Evaluation metrics have been intentionally excluded.
The focus of this version is to demonstrate a functioning AI documentation pipeline, not model benchmarking.
Future iterations will include validation on real-world doctor-patient datasets.

🧰 Installation
Step 1: Clone Repository
bash
Copy code
git clone https://github.com/yourusername/Physician-Notetaker.git
cd Physician-Notetaker
Step 2: Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
Step 3: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: Download Models
bash
Copy code
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_md
python -m spacy validate
▶️ Usage
Run Full Pipeline
bash
Copy code
python src/pipeline_runner.py --input data/sample_transcript.txt
Output:

output/ner_results.json – Extracted entities

output/summary.json – Medical summary

output/sentiment_results.json – Sentiment/intent

output/soap_note.json – Final SOAP note

Run Individual Components
bash
Copy code
# 1. NER Extraction
python src/ner_extractor.py --input data/sample_transcript.txt

# 2. Summarization
python src/summarizer.py --transcript data/sample_transcript.txt --ner_output output/ner_results.json

# 3. Sentiment Analysis
python src/sentiment_analyzer.py --input data/sample_transcript.txt

# 4. SOAP Generation
python src/soap_generator.py --ner output/ner_results.json --summary output/summary.json --sentiment output/sentiment_results.json
📁 Project Structure
pgsql
Copy code
Physician-Notetaker/
├── data/
│   └── sample_transcript.txt
├── src/
│   ├── ner_extractor.py
│   ├── summarizer.py
│   ├── sentiment_analyzer.py
│   ├── soap_generator.py
│   └── pipeline_runner.py
├── output/
│   ├── ner_results.json
│   ├── summary.json
│   ├── sentiment_results.json
│   └── soap_note.json
├── requirements.txt
└── README.md
⚠️ Limitations
Current models trained/fine-tuned on synthetic or limited data

Rule-based SOAP generation may lack flexibility for complex cases

Real-world validation required for robust deployment
