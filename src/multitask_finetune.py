

import json
from transformers import pipeline

# Initialize pretrained zero-shot model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define classification labels
sentiment_labels = ["Anxious", "Neutral", "Reassured"]
intent_labels = [
    "Reporting symptoms",
    "Seeking reassurance",
    "Expressing concern",
    "Confirming improvement"
]

def analyze_sentiment_intent(text):
    """Predict sentiment and intent using pretrained zero-shot model."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"Sentiment": "Unknown", "Intent": "Unknown"}

    # Sentiment prediction
    sentiment_result = classifier(text, sentiment_labels)
    sentiment = sentiment_result["labels"][0]

    # Intent prediction
    intent_result = classifier(text, intent_labels)
    intent = intent_result["labels"][0]

    return {"Sentiment": sentiment, "Intent": intent}


def run_pipeline(
    input_path="data/processed/patient_text.txt",
    output_path="data/processed/sentiment_results.json"
):
    """Run sentiment & intent prediction for patient transcript."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print("🔍 Processing text for sentiment and intent detection...\n")
    result = analyze_sentiment_intent(text)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Predictions saved to: {output_path}")
    print("\n🧾 Output:\n", json.dumps(result, indent=4))


if __name__ == "__main__":
    run_pipeline()
