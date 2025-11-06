#  Physician Notetaker – AI Medical Documentation System

An end-to-end AI pipeline that automatically converts doctor-patient conversations into structured **SOAP notes** using NLP and transformer-based models.

---

##  Table of Contents

1. [Overview](#overview)  
2. [System Architecture](#system-architecture)  
3. [Key Components](#key-components)  
4. [Tech Stack](#tech-stack)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Results](#results)

---

##  Overview

**Physician Notetaker** automates medical documentation by processing raw doctor-patient transcripts and generating structured SOAP notes. The system uses a hybrid approach combining BioBERT transformers with rule-based NLP methods.

### Key Features

- **Hybrid NER System** – 86% F1-score for medical entity extraction
- **Multi-Task Learning** – Joint sentiment and intent classification  
- **Auto SOAP Notes** – Structured clinical documentation
- **Modular Pipeline** – Independent, reusable components

---

##  System Architecture

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
```

---

##  Key Components

### 1. Named Entity Recognition (NER)
- **BioBERT Transformer** – Pre-trained on 470K+ medical papers
- **scispaCy** – Rule-based pattern matching with 100K+ medical terms
- **Regex Patterns** – Extract measurements, dates, and quantities

**Output:** Symptoms, Diagnosis, Treatment, Prognosis


<img width="1191" height="1017" alt="image" src="https://github.com/user-attachments/assets/11bced6d-2280-4cc4-bfeb-5f2d9c35b3a7" />


### 2. Medical Summary Generation
- Deduplicates and cleans extracted entities
- Extracts patient information using regex
- Maps entities to medical categories

**Output:** Structured JSON summary


<img width="1795" height="878" alt="image" src="https://github.com/user-attachments/assets/b38ab9b8-a1d9-47c0-a882-f4c53031da7e" />


### 3. Sentiment & Intent Analysis
- **Multi-task BioBERT** classifier with two heads:
  - **Sentiment:** Anxious / Neutral / Reassured
  - **Intent:** Reporting / Seeking / Expressing / Confirming
- Fine-tuned on 3,000 medical conversation samples

**Output:** Patient emotional state and conversation intent

<img width="1343" height="408" alt="image" src="https://github.com/user-attachments/assets/e4ac2842-b445-4854-b39c-9bcadb10d76c" />


### 4. SOAP Note Generation
- **S** (Subjective): Patient complaints & history
- **O** (Objective): Doctor observations & exam findings
- **A** (Assessment): Diagnosis & clinical impression
- **P** (Plan): Treatment & follow-up recommendations

**Output:** Clinical SOAP note in JSON format

<img width="1878" height="1013" alt="image" src="https://github.com/user-attachments/assets/57666edc-4fd6-47fc-96ff-876aea7417b5" />


---

##  Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| NER | BioBERT (`d4data/biomedical-ner-all`) | Medical entity extraction |
| Secondary NER | scispaCy (`en_core_sci_md`) | Rule-based patterns |
| Sentiment/Intent | Fine-tuned BioBERT | Emotion & intent classification |
| Framework | HuggingFace Transformers, PyTorch | Deep learning |
| NLP Tools | spaCy 3.7+ | Text processing |

---

##  Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Physician-Notetaker.git
cd Physician-Notetaker
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Models
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_md
python -m spacy validate
```

---

##  Usage

### Run Full Pipeline
```bash
python src/pipeline_runner.py --input data/sample_transcript.txt
```

**Output:**
- `output/ner_results.json` – Extracted entities
- `output/summary.json` – Medical summary
- `output/sentiment_results.json` – Sentiment/intent
- `output/soap_note.json` – Final SOAP note

### Run Individual Components

```bash
# 1. NER Extraction
python src/ner_extractor.py --input data/sample_transcript.txt

# 2. Summarization
python src/summarizer.py --transcript data/sample_transcript.txt --ner_output output/ner_results.json

# 3. Sentiment Analysis
python src/sentiment_analyzer.py --input data/sample_transcript.txt

# 4. SOAP Generation
python src/soap_generator.py --ner output/ner_results.json --summary output/summary.json --sentiment output/sentiment_results.json
```

---

##  Results

### Performance Metrics

| Component | Metric | Score |
|-----------|--------|-------|

| Sentiment | Accuracy | 1.00* |
| Intent | Accuracy | 1.00* |

*100% accuracy on synthetic test data (potential overfitting)


##  Project Structure

```
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
```

---

##  Limitations

- Trained on synthetic data – needs real-world validation
- Rule-based SOAP generation lacks flexibility
- Potential overfitting in sentiment/intent model
