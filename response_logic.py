from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Path to the locally saved fine-tuned model
model_path = "./fine_tuned_model"

# Check if the model directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path '{model_path}' not found. Please ensure the model is saved in this directory.")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a text classification pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# Emotion label mapping from model config
id2label = model.config.id2label

# Get the predicted emotion
def get_emotion(text):
    if not text or not text.strip():
        return "neutral"  # Default if input is empty

    scores = classifier(text)[0]
    top = max(scores, key=lambda x: x["score"])
    return top["label"]

# Generate responses based on emotion
def get_response(emotion):
    emotion_lower = emotion.lower()
    responses = {
        "joy": "That's wonderful to hear! 😊 Stay positive!",
        "sadness": "I'm sorry you're feeling this way. You're not alone. 💙",
        "anger": "It's okay to feel angry sometimes. Take a deep breath. 🧘",
        "fear": "I understand your concern. You're stronger than you think.",
        "surprise": "Wow! That must’ve been unexpected!",
        "love": "That’s so sweet! Keep spreading love. 💖",
        "neutral": "Thanks for sharing. I’m here if you want to talk more.",
        "anxiety": "It’s okay to feel anxious. Try to take it one step at a time.",
        "stress": "That sounds stressful. Don’t forget to take care of yourself.",
    }
    return responses.get(emotion_lower, "I'm here to listen. Tell me more about how you're feeling.")
