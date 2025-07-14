import streamlit as st
import sys
import os
import time # For simulation of loading time if needed, or for general time.sleep()

# --- CRITICAL: Ensure your 'response_logic.py' is accessible ---
# This line adds the parent directory of the current script to Python's search path.
# Adjust this path based on your actual file structure.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from response_logic import get_emotion, get_response
except ImportError:
    st.error("Error: Could not import 'get_emotion' or 'get_response' from 'response_logic.py'.")
    st.error("Please ensure 'response_logic.py' is in the correct location and contains these functions.")
    st.stop() # Stop the app if crucial functions aren't found


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Emotional Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "## Emotion Chatbot\nThis chatbot uses a fine-tuned Hugging Face Transformers model to understand and respond to your emotions."
    }
)

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    /* Main Title Styling */
    .st-emotion-chat-title {
        font-size: 3.5em;
        font-weight: bold;
        color: #4CAF50; /* A vibrant green */
        text-align: center;
        margin-bottom: 0.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    /* Subheader Styling */
    .st-emotion-chat-subheader {
        font-size: 1.5em;
        color: #555555;
        text-align: center;
        margin-bottom: 1.5em;
        font-style: italic;
    }
    /* Button Styling */
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 1.2em;
        font-weight: bold;
        transition-duration: 0.4s;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
        color: white;
        transform: translateY(-2px); /* Slight lift effect */
    }
    /* Text Area Label Styling */
    .stTextArea label {
        font-size: 1.2em;
        font-weight: bold;
        color: #333333;
    }
    /* Chat Message Container */
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    .chat-message.user {
        justify-content: flex-end;
    }
    .chat-message.bot {
        justify-content: flex-start;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #eee;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5em;
        margin: 0 10px;
        flex-shrink: 0;
    }
    .chat-message.user .chat-avatar {
        order: 2; /* User avatar on the right */
        background-color: #e0f7fa; /* Light blue */
    }
    .chat-message.bot .chat-avatar {
        order: 1; /* Bot avatar on the left */
        background-color: #ffe0b2; /* Light orange */
    }
    .chat-bubble {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 20px;
        line-height: 1.4;
        position: relative;
    }
    .chat-message.user .chat-bubble {
        background-color: #DCF8C6; /* Light green for user */
        margin-left: auto; /* Push to right */
        border-bottom-right-radius: 5px; /* Triangle effect */
    }
    .chat-message.bot .chat-bubble {
        background-color: #E0E0E0; /* Light gray for bot */
        margin-right: auto; /* Push to left */
        border-bottom-left-radius: 5px; /* Triangle effect */
    }
    .chat-bubble-emotion {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Section ---
st.markdown('<p class="st-emotion-chat-title">🤖 Emotional Sentiment Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="st-emotion-chat-subheader">Talk to me! I’ll understand how you\'re feeling 💬</p>', unsafe_allow_html=True)

# --- About Section (Collapsible for effectiveness) ---
with st.expander("✨ About This Chatbot & How It Works"):
    st.write(
        """
        This chatbot utilizes a fine-tuned **Hugging Face Transformers model** to analyze the sentiment and
        detect the underlying emotion in your text. Based on the detected emotion, it provides an empathetic and
        relevant response.

        **How it works:**
        1.  Type your thoughts or feelings into the text area below.
        2.  Click the "Send Message" button.
        3.  The bot processes your input to determine the emotion (e.g., joy, sadness, anger, etc.).
        4.  It then generates a personalized empathetic response based on the detected emotion.
        """
    )
    st.info("💡 **Tip:** Try expressing a range of emotions to see how the bot reacts!")

# --- Initialize session state for chat history and input ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input_content" not in st.session_state:
    st.session_state.user_input_content = ""
# Flag to indicate if model is loading for the first time in this session
if "model_loaded_once" not in st.session_state:
    st.session_state.model_loaded_once = False


# --- Callback function to start a new chat ---
def start_new_chat():
    st.session_state.messages = [] # Clear messages history
    st.session_state.user_input_content = "" # Clear input box
    st.rerun() # Force a rerun to update the UI


# --- Display Chat History ---
chat_history_container = st.container(height=400, border=True) # Scrollable container for history

with chat_history_container:
    # Display initial welcome message if no history
    if not st.session_state.messages:
        st.info("👋 Hello! Type a message below to start chatting with the Emotional Chatbot.")

    # Iterate and display messages
    for message in st.session_state.messages:
        avatar = "😊" if message["role"] == "user" else "🤖"
        message_class = "user" if message["role"] == "user" else "bot"

        with st.container():
            st.markdown(f"""
                <div class="chat-message {message_class}">
                    <div class="chat-avatar">{avatar}</div>
                    <div class="chat-bubble">
                        {message["content"]}
                        {'<div class="chat-bubble-emotion">Detected: ' + message["emotion"].capitalize() + '</div>' if message["role"] == "bot" else ''}
                    </div>
                </div>
            """, unsafe_allow_html=True)


# --- Chat Input and Buttons ---
user_input = st.text_area(
    "Your Message:",
    value=st.session_state.user_input_content,
    placeholder="Type your thoughts here, like 'I'm feeling really happy today!' or 'This news made me so sad.'",
    height=80,
    key="user_input_area"
)

col1, col2, col3 = st.columns([1, 1, 3]) # Adjust column ratios if needed

with col1:
    send_button = st.button("Send Message ✨")
with col2:
    st.button("New Chat 🔄", on_click=start_new_chat)


# --- Send Button Action ---
if send_button:
    # Update the session state variable with the current text area content.
    st.session_state.user_input_content = user_input

    if user_input.strip():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input, "emotion": None})

        # Display initial loading message only if model hasn't been loaded this session
        if not st.session_state.model_loaded_once:
            st.info("⏳ First-time analysis might take a moment as the emotion model loads...")

        with st.spinner("🧠 Analyzing your emotions..."):
            try:
                # Get emotion and response
                emotion = get_emotion(user_input)
                bot_response = get_response(emotion)

                # Set flag indicating model has loaded at least once
                st.session_state.model_loaded_once = True

                # Add bot response to history
                st.session_state.messages.append({
                    "role": "bot",
                    "content": bot_response,
                    "emotion": emotion # Store emotion for display or other logic
                })

                # Trigger balloons for joy!
                if emotion.lower() == 'joy':
                    st.balloons()

                # Clear the input text area after sending
                st.session_state.user_input_content = ""
                st.rerun() # Rerun to update chat history and clear input

            except Exception as e:
                st.error(f"An error occurred during emotion analysis: {e}")
                st.warning("Please ensure your 'fine_tuned_model' is correctly set up and accessible by 'response_logic.py'.")
                # Add an error message to chat history if needed
                st.session_state.messages.append({"role": "bot", "content": f"Oops! I encountered an error: {e}", "emotion": "error"})
                st.rerun() # Rerun to show the error message in chat history
    else:
        st.warning("Please type something into the text area before clicking 'Send Message'.")

# --- Footer (Optional) ---
st.markdown("---")
st.markdown("Created with ❤️ by your AI companion using Streamlit and Hugging Face Transformers.")
st.markdown("For best results, ensure your fine-tuned model is properly loaded.")