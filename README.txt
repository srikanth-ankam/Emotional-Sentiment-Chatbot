🤖 Emotion-Aware Chatbot: Sentiment Analysis + Adaptive Response System

This project is a prototype chatbot that identifies the emotional state of a user based on conversational input and responds with culturally empathetic and context-aware replies.



🧠 Project Overview

- Objective: Build a chatbot that performs **emotional sentiment analysis** and **generates supportive, culturally relevant responses**.
- Use Case: Can be applied in mental health support, wellness chatbots, and emotionally intelligent interfaces.

📂 Folder Structure

emotion-chatbot/
├── app.py # Streamlit UI for the chatbot
├── response_logic.py # Emotion detection and response generation logic
├── evaluate_chatbot.py # Script to evaluate emotion prediction accuracy
├── extended.csv # Cleaned dataset 
├── fine_tuned_model -config.json
                      -model.safetensors
                      -special_tokens_map
                      -tokenizer
                      -tokenizer_config
                      -vocab


🧪 Features

- Detects emotions: `joy`, `sadness`, `anxiety`, `stress`, etc.
- Uses HuggingFace Transformers to classify emotional state.
- Responds with pre-defined empathetic replies tailored to each emotion.
- Evaluated using real test cases with overall performance accuracy.
- Culturally adaptive — ideal for Indian English expressions.

🚀 How to Run

1. 🛠 Install Dependencies

pip install -r requirements.txt

2. ▶️ Launch the Chatbot

streamlit run app.py

3. 🧪 Run Evaluation

python evaluate_chatbot.py

📊 Dataset
extended.csv: 300000+ manually labeled sentences with emotional categories.

Used for both training and testing.

📈 Evaluation Result
Test Accuracy: 77.15% using the base model.

Insights: Model struggles with nuanced or culturally specific expressions. Accuracy can be improved via fine-tuning.

💬 Sample Emotions & Responses

Emotion	             Example User Input	                                                    Chatbot Response
joy	             "I just got a job!"	                         "That's wonderful! Congratulations on your success!"
sadness	             "I feel empty lately."	                         "I'm here for you. It's okay to feel low sometimes."
stress	             "Everything's piling up at work."	                 "That sounds overwhelming. Try to take small breaks when you can."
anxiety	             "I'm so nervous about my exams."	                 "You're not alone in this. Just do your best, one step at a time."




