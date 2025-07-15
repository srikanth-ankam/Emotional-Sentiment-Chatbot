from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def get_emotion(text):
    if not text or not text.strip():
        return "neutral", 1.0

    raw_scores = classifier(text, return_all_scores=True)[0]

    top = max(raw_scores, key=lambda x: x["score"])
    return top["label"], float(top["score"])

def get_response(emotion):
    emotion = emotion.lower()
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
    return responses.get(emotion, "I'm here to listen. Tell me more about how you're feeling.")
