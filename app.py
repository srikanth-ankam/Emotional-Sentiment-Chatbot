# app.py

import streamlit as st
from response_logic import get_emotion, get_response

st.title("😊 Emotion-Aware Chatbot")
user_input = st.text_area("Enter a message:")

if st.button("Send"):
    if user_input.strip():
        emotion, confidence = get_emotion(user_input)
        response = get_response(emotion)

        st.write(f"**Emotion Detected:** {emotion} (confidence: {confidence:.1%})")
        st.write(f"**Bot Response:** {response}")
    else:
        st.warning("Please enter a message to analyze.")
