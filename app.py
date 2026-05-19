import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ======================
# DOWNLOAD STOPWORDS
# ======================
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ======================
# DARK THEME CSS
# ======================
def apply_dark_theme():
    st.markdown("""
        <style>
            /* Main background */
            .stApp {
                background-color: #0e1117;
                color: #e0e0e0;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #161b22;
                color: #e0e0e0;
            }
            section[data-testid="stSidebar"] * {
                color: #e0e0e0 !important;
            }

            /* Title and text */
            h1, h2, h3, h4, h5, h6, p, label {
                color: #e0e0e0 !important;
            }

            /* Chat input box */
            .stChatInput textarea {
                background-color: #1c2128 !important;
                color: #e0e0e0 !important;
                border: 1px solid #30363d !important;
                border-radius: 10px !important;
            }

            /* Chat input container */
            .stChatInput {
                background-color: #0e1117 !important;
                border-top: 1px solid #30363d;
            }

            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #0e1117;
            }
            ::-webkit-scrollbar-thumb {
                background: #30363d;
                border-radius: 10px;
            }

            /* Buttons */
            .stButton > button {
                background-color: #21262d !important;
                color: #e0e0e0 !important;
                border: 1px solid #30363d !important;
                border-radius: 8px !important;
                transition: background 0.2s;
            }
            .stButton > button:hover {
                background-color: #30363d !important;
                border-color: #58a6ff !important;
            }

            /* Bar chart area */
            .stVegaLiteChart, .stPlotlyChart {
                background-color: #161b22 !important;
                border-radius: 10px;
            }

            /* Divider */
            hr {
                border-color: #30363d;
            }

            /* User bubble */
            .user-bubble {
                text-align: right;
                background: #1f4e3d;
                padding: 10px 14px;
                border-radius: 16px 16px 4px 16px;
                margin: 6px 0;
                display: inline-block;
                float: right;
                clear: both;
                max-width: 75%;
                color: #d1f7e0;
                font-size: 15px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            }

            /* Bot bubble */
            .bot-bubble {
                text-align: left;
                background: #1c2128;
                padding: 10px 14px;
                border-radius: 16px 16px 16px 4px;
                margin: 6px 0;
                display: inline-block;
                float: left;
                clear: both;
                max-width: 75%;
                color: #c9d1d9;
                font-size: 15px;
                border-left: 3px solid #58a6ff;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            }

            /* Emotion tag */
            .emotion-tag {
                font-size: 11px;
                color: #8b949e;
                font-style: italic;
            }
        </style>
    """, unsafe_allow_html=True)

# ======================
# PREPROCESS FUNCTION
# ======================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ======================
# LOAD MODEL + DATA
# ======================
@st.cache_resource
def load_artifacts():
    with open('emotion_model2.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('tfidf_vectorizer2.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    loaded_conv_df = pd.read_csv('mental_health_conversations.csv')
    loaded_conv_df['clean_question'] = loaded_conv_df['question'].astype(str).apply(preprocess)
    loaded_X_conv = loaded_vectorizer.transform(loaded_conv_df['clean_question'])

    return loaded_model, loaded_vectorizer, loaded_conv_df, loaded_X_conv

model, vectorizer, conv_df, X_conv = load_artifacts()

# ======================
# BUILD CONTEXT FROM HISTORY
# ======================
def build_context(history, max_turns=5):
    recent = history[-(max_turns * 2):]
    context_parts = []
    for sender, msg in recent:
        prefix = "User" if sender == "You" else "Bot"
        context_parts.append(f"{prefix}: {msg}")
    return " ".join(context_parts)

# ======================
# RESPONSE FUNCTION
# ======================
def get_therapist_response(user_input, context=""):
    combined_text = f"{context} {user_input}".strip() if context else user_input
    user_clean = preprocess(combined_text)
    user_vec = vectorizer.transform([user_clean])

    similarity = cosine_similarity(user_vec, X_conv)
    best_index = np.argmax(similarity)
    best_score = similarity[0][best_index]

    if best_score < 0.1:
        return "I hear you. Could you tell me more about how you're feeling?"

    return conv_df['answer'].iloc[best_index]

# ======================
# EMOTION WITH CONTEXT
# ======================
def detect_emotion_with_context(user_input, context=""):
    combined_text = f"{context} {user_input}".strip() if context else user_input
    clean = preprocess(combined_text)
    emotion = model.predict(vectorizer.transform([clean]))[0]
    return emotion

# ======================
# CHATBOT FUNCTION
# ======================
def chatbot(user_input, history):
    context = build_context(history, max_turns=5)
    emotion = detect_emotion_with_context(user_input, context)
    response = get_therapist_response(user_input, context)
    return emotion, response

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="centered")
apply_dark_theme()

st.title("🧠 Mental Health Chatbot")
st.markdown("<p style='color:#8b949e;'>Hello! I'm here to listen. How are you feeling today?</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# ---- Sidebar Emotion Tracker ----
with st.sidebar:
    st.markdown("## 📊 Emotion Tracker")
    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.emotion_history:
        emotion_counts = pd.Series(st.session_state.emotion_history).value_counts()
        st.bar_chart(emotion_counts)
        st.markdown(
            f"<p style='color:#58a6ff;'>Last detected: <b>{st.session_state.emotion_history[-1]}</b></p>",
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='color:#8b949e;'>Total messages: {len(st.session_state.emotion_history)}</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#8b949e;'>No emotions tracked yet.</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation"):
        st.session_state.history = []
        st.session_state.emotion_history = []
        st.rerun()

# ---- Chat Display ----
chat_container = st.container()
with chat_container:
    for i, (sender, msg) in enumerate(st.session_state.history):
        if sender == "You":
            st.markdown(
                f"<div class='user-bubble'>🧑 <b>You:</b> {msg}</div>",
                unsafe_allow_html=True
            )
        else:
            bot_turn_index = sum(1 for s, _ in st.session_state.history[:i] if s == "Bot")
            emotion_label = ""
            if bot_turn_index < len(st.session_state.emotion_history):
                emotion_label = f"<br><span class='emotion-tag'>Emotion: {st.session_state.emotion_history[bot_turn_index]}</span>"

            st.markdown(
                f"<div class='bot-bubble'>🤖 <b>Therapist:</b> {msg}{emotion_label}</div>",
                unsafe_allow_html=True
            )

st.markdown("<div style='clear:both; margin-bottom:80px;'></div>", unsafe_allow_html=True)

# ---- Input ----
user_input = st.chat_input("Type your message here...")

if user_input:
    emotion, response = chatbot(user_input, st.session_state.history)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))
    st.session_state.emotion_history.append(emotion)
    st.rerun()