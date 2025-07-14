from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load local fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a sentiment analysis pipeline using the fine-tuned model
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# Emotion label mapping
id2label = model.config.id2label

# Get the predicted emotion
def get_emotion(text):
    scores = classifier(text)[0]
    top = max(scores, key=lambda x: x["score"])
    return top["label"]

# Generate empathetic responses
def get_response(emotion):
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
    return responses.get(emotion.lower(), "I'm here to listen. Tell me more about how you're feeling.")
