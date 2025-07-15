<h2>🤖 Emotion-Aware Chatbot: Sentiment Analysis + Adaptive Response System</h2>

This project is a prototype chatbot that identifies the emotional state of a user based on conversational input and responds with culturally empathetic and context-aware replies.



<h2>🧠 Project Overview</h2>

- Objective: Build a chatbot that performs **emotional sentiment analysis** and **generates supportive, culturally relevant responses**.
- Use Case: Can be applied in mental health support, wellness chatbots, and emotionally intelligent interfaces.


<h2>🚀 Live Streamlit Demo (Emotion-Aware Chatbot): </h2>
https://emotional-sentiment-chatbot.streamlit.app


<h2>📂 Folder Structure</h2>

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


<h2>🧪 Features</h2>

- Detects emotions: `joy`, `sadness`, `anxiety`, `stress`, etc.
- Uses HuggingFace Transformers to classify emotional state.
- Responds with pre-defined empathetic replies tailored to each emotion.
- Evaluated using real test cases with overall performance accuracy.
- Culturally adaptive — ideal for Indian English expressions.

<h2>🚀 How to Run</h2>

1. 🛠 Install Dependencies

pip install -r requirements.txt

2. ▶️ Launch the Chatbot

streamlit run app.py

3. 🧪 Run Evaluation

python evaluate_chatbot.py

<h2>📊 Dataset</h2>
extended.csv: 300000+ manually labeled sentences with emotional categories.

Used for both training and testing.

<h2>📈 Evaluation Result</h2>
Test Accuracy: 77.15% using the base model.

Insights: Model struggles with nuanced or culturally specific expressions. Accuracy can be improved via fine-tuning.

<h2>💬 Sample Emotions & Responses</h2>

Emotion	                                 Example User Input	                                                              Chatbot Response
joy	                                   "I just got a job!"	                                                              "That's wonderful! Congratulations on your success!"
sadness	                               "I feel empty lately."	                                                            "I'm here for you. It's okay to feel low sometimes."
stress	                               "Everything's piling up at work."	                                                "That sounds overwhelming. Try to take small breaks when you can."
anxiety	                               "I'm so nervous about my exams."	                                                  "You're not alone in this. Just do your best, one step at a time."

<h2>Screenshots</h2>

<img width="1202" height="776" alt="Screenshot 2025-07-14 195447" src="https://github.com/user-attachments/assets/e48e489d-5408-4e83-9630-0b4575bf4d5f" />
<img width="1240" height="819" alt="Screenshot 2025-07-14 195428" src="https://github.com/user-attachments/assets/17baba47-5e7d-4402-9423-aa6b0dc70fcb" />
<img width="1179" height="620" alt="Screenshot 2025-07-14 195407" src="https://github.com/user-attachments/assets/bd90ce8e-6abf-4e9a-ba87-1489125b6000" />

