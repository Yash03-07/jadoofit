import os
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Create a .env file from .env.example and add your key.")
    st.stop()

MODEL_NAME = "gemini-2.0-flash"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

KNOWLEDGE_FILE = Path(__file__).with_name("fitness_knowledge.txt")
SYSTEM_PROMPT = (
    "You are a positive, beginner-friendly fitness and health coach. "
    "Give safe, encouraging advice focused on workouts, nutrition, recovery, sleep, and healthy habits. "
    "If the user asks something outside fitness and health, reply: 'I’m a fitness coach and can answer questions related to that.'"
)

MEMORY_LIMIT = 6


@st.cache_data
def load_knowledge_base(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


KNOWLEDGE_BASE = load_knowledge_base(KNOWLEDGE_FILE)


def retrieve_context(question: str, knowledge_base, top_k=3):
    q = question.lower().replace("?", " ")
    words = [w for w in q.split() if len(w) > 2]

    scored = []
    for item in knowledge_base:
        text = item.lower()
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def build_prompt(user_input: str, chat_memory):
    context = retrieve_context(user_input, KNOWLEDGE_BASE)
    context_text = "\n".join(f"- {item}" for item in context) if context else "No extra knowledge found."

    history_text = ""
    for role, text in chat_memory[-MEMORY_LIMIT:]:
        history_text += f"{role}: {text}\n"

    prompt = f"{SYSTEM_PROMPT}\n\n"
    prompt += "Use the following fitness knowledge if relevant:\n" + context_text + "\n\n"
    prompt += "Recent conversation memory:\n" + history_text + "\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt


def fallback_answer(user_input: str, chat_memory):
    context = retrieve_context(user_input, KNOWLEDGE_BASE)
    context_text = "\n".join(f"- {item}" for item in context) if context else "- Stay consistent with simple movement, hydration, and rest."
    memory_text = "\n".join(f"{role}: {text}" for role, text in chat_memory[-MEMORY_LIMIT:])
    memory_note = f"\nRecent memory:\n{memory_text}" if memory_text else ""
    return (
        "I’m using the local fitness guidance available right now. "
        "Here are the most relevant tips:\n"
        f"{context_text}{memory_note}\n"
        "If you want a full AI answer, please try again after your Gemini quota resets."
    )


st.set_page_config(page_title="JadooFit Chatbot", page_icon="💪", layout="centered")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background-color: #0b1220; color: #eff6ff; }
    [data-testid="stSidebar"] { background-color: #111827; }
    .stButton > button { border-radius: 999px; background: linear-gradient(135deg, #22c55e, #14b8a6); color: white; border: none; }
    .stButton > button:hover { background: linear-gradient(135deg, #16a34a, #0f766e); color: white; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💪 JadooFit Coach")
st.caption("A polished, beginner-friendly fitness and health chatbot with memory and simple RAG guidance.")

col1, col2 = st.columns([1, 1])
with col1:
    st.info("Try: beginner workout plan, protein ideas, recovery tips, or healthy sleep habits.")
with col2:
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.chat_memory = []
        st.rerun()

quick_prompts = [
    "Give me a simple beginner workout plan for 3 days",
    "What should I eat for better energy?",
    "How can I improve my sleep and recovery?",
]

button_row = st.columns(len(quick_prompts))
for i, prompt in enumerate(quick_prompts):
    with button_row[i]:
        if st.button(prompt, use_container_width=True):
            st.session_state.chat_memory.append(("User", prompt))
            try:
                answer = model.generate_content(build_prompt(prompt, st.session_state.chat_memory)).text.strip()
            except Exception:
                answer = fallback_answer(prompt, st.session_state.chat_memory)
            st.session_state.chat_memory.append(("Assistant", answer))
            st.rerun()

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

for role, text in st.session_state.chat_memory:
    with st.chat_message(role.lower() if role != "Assistant" else "assistant"):
        st.write(text)

user_input = st.chat_input("Ask about workouts, nutrition, sleep, or recovery")

if user_input:
    st.session_state.chat_memory.append(("User", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    try:
        prompt = build_prompt(user_input, st.session_state.chat_memory)
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as exc:
        answer = fallback_answer(user_input, st.session_state.chat_memory)

    st.session_state.chat_memory.append(("Assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)
