import os
import numpy as np
import pandas as pd
import random
import torch
import streamlit as st
import pyttsx3
import speech_recognition as sr
import threading
from datetime import datetime
from docx import Document

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Streamlit App Title
st.title("Vireum AI Assistant (RAG + Voice)")
st.write("Ask about your AI project — strictly using document knowledge.")

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

EMBED_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.npy"

# Load .docx RAG documents
def load_documents():
    folder = "docs"
    texts = []
    if not os.path.exists(folder):
        st.error("Docs folder not found!")
        return texts
    for file in os.listdir(folder):
        if file.endswith(".docx"):
            path = os.path.join(folder, file)
            doc = Document(path)
            full_text = "\n".join([p.text for p in doc.paragraphs])
            texts.append(full_text)
    return texts

# Chunking function
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Load documents and create chunks
raw_documents = load_documents()
chunks = []
for doc in raw_documents:
    chunks.extend(chunk_text(doc))

if not chunks:
    st.warning("No text chunks found in documents!")

# Create or load embeddings
def create_or_load_embeddings(chunks):
    if os.path.exists(EMBED_FILE) and os.path.exists(CHUNKS_FILE):
        embeddings = np.load(EMBED_FILE)
        chunks_loaded = np.load(CHUNKS_FILE, allow_pickle=True)
        if embeddings.size == 0 or len(chunks_loaded) == 0:
            st.warning("Embeddings or chunks empty, regenerating...")
        else:
            st.success("Loaded cached embeddings")
            return embeddings, chunks_loaded

    st.info("Generating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    np.save(EMBED_FILE, embeddings)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))
    st.success("Embeddings ready!")
    return embeddings, chunks

embeddings, chunks = create_or_load_embeddings(chunks)

# Load LLM 
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"  
    pipe = pipeline(
        "text2text-generation",   
        model=model_name,
        tokenizer=model_name,
        device_map="cpu",
    )
    return pipe

llm = load_llm()

# Vector search
def search(query, embeddings, chunks, top_k=1):
    if len(embeddings) == 0 or len(chunks) == 0:
        st.error("No embeddings or chunks available for search.")
        return "No relevant document found."
    query_vec = embedder.encode([query])[0]
    scores = np.dot(embeddings, query_vec)
    best = np.argmax(scores)
    return chunks[best]

# Voice input + output
recognizer = sr.Recognizer()

def listen():
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        return recognizer.recognize_google(audio)
    except:
        return ""

def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()


# UI Inputs
name = st.text_input("Your Name")
email = st.text_input("Your Email")
project = st.text_area("Your project idea")

if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""

if st.button("Voice Input"):
    voice = listen()
    if voice:
        st.session_state.voice_input = voice
        project += " " + voice
        st.text_area("Updated Project:", value=project)

# Load CSV
DATA_FILE = "data/users.csv"
os.makedirs("data", exist_ok=True)

try:
    users = pd.read_csv(DATA_FILE)
except:
    users = pd.DataFrame(columns=["timestamp","name","email","project","voice_input","estimate","schedule"])

# Main Estimate Button
if st.button("Get Estimate"):
    if not (name and email and project):
        st.warning("Fill all fields first")
    else:
        # Retrieve relevant chunk
        retrieved = search(project, embeddings, chunks)
        retrieved = retrieved[:1500]  # truncate for faster CPU inference

        # Strict RAG prompt
        prompt = f"""
You are an AI assistant.
Answer the user's question STRICTLY using ONLY the information in <context>.
If the answer is not found in the context, reply exactly with:
"I don’t have information about that."

<context>
{retrieved}
</context>

User question: {project}

Answer:
"""

        # LLM response
        response = llm(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            #return_full_text=True,
            clean_up_tokenization_spaces=True
        )
        suggestion = response[0]["generated_text"]

        # Estimate calculation
        words = len(project.split())
        cost = round(500 + words * random.uniform(2, 4), 2)
        weeks = random.randint(2, 8)

        st.success(f"Estimated Cost: €{cost} — Duration: {weeks} weeks")
        st.info("AI Suggestion:\n" + suggestion)

        speak(f"Your estimated cost is {cost} euros. Project duration {weeks} weeks.")

        # Save user entry
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "email": email,
            "project": project,
            "voice_input": st.session_state.voice_input,
            "estimate": cost,
            "schedule": f"{weeks} weeks"
        }

        users = pd.concat([users, pd.DataFrame([entry])])
        users.to_csv(DATA_FILE, index=False)
        st.success("Saved!")

# Show Saved Entries
st.markdown("---")
if st.button("Show Saved"):
    st.dataframe(users)
