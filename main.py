import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Create a .env file from .env.example and add your key.")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

KNOWLEDGE_FILE = Path(__file__).with_name("fitness_knowledge.txt")

SYSTEM_PROMPT = (
    "You are a positive, beginner-friendly fitness and health coach. "
    "Give safe, encouraging advice focused on workouts, nutrition, recovery, sleep, and healthy habits. "
    "If the user asks something outside fitness and health, reply: 'I’m a fitness coach and can answer questions related to that.'"
)

MEMORY_LIMIT = 6
chat_memory = []


def load_knowledge_base(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


KNOWLEDGE_BASE = load_knowledge_base(KNOWLEDGE_FILE)


def retrieve_context(question: str, knowledge_base, top_k=3):
    """Simple keyword-based RAG retrieval from a local fitness knowledge file."""
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


def build_prompt(user_input: str):
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


def fallback_answer(user_input: str):
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


print("jadoofitbot: Hello! I’m your fitness coach. I can help with workouts, nutrition, recovery, and healthy habits.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("Bot: Goodbye!")
        break

    try:
        prompt = build_prompt(user_input)
        response = model.generate_content(prompt)
        answer = response.text.strip()

        chat_memory.append(("User", user_input))
        chat_memory.append(("Assistant", answer))

        print("Bot:", answer)
    except Exception as exc:
        answer = fallback_answer(user_input)
        chat_memory.append(("User", user_input))
        chat_memory.append(("Assistant", answer))
        print("Bot:", answer)
        print("(Local fallback used because Gemini returned:", exc, ")")
