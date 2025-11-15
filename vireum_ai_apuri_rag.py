# ========================
# RAG Setup
# ========================
import os
import streamlit as st
import pandas as pd
import torch
import threading
import pyttsx3
import speech_recognition as sr
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import datetime as dt
from datetime import datetime
#from google_calendar import create_event
import random


# Load PDF
def load_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_text(text, max_len=500):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_len:
            current_chunk += para + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Load embeddings model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_embeddings(chunks, file="vireum_embeddings.npy"):
    if os.path.exists(file):
        print("Loading embeddings...")
        return np.load(file, allow_pickle=True)
    else:
        print("Creating embeddings...")
        embeddings = embed_model.encode(chunks)
        np.save(file, embeddings)
        return embeddings

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve(query, index, chunks, top_k=3):
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

# Load PDF and embeddings
pdf_text = load_pdf_text("Vireum_Vision_AI.pdf")
chunks = split_text(pdf_text)
embeddings = create_embeddings(chunks)
index = build_index(embeddings)

# ========================
# Chatbot function with RAG
# ========================
def generate_rag_response(user_input):
    retrieved_chunks = retrieve(user_input, index, chunks)
    context_text = "\n".join(retrieved_chunks)
    prompt = f"""
You are Vireum AI Assistant. Use the following context from Vireum Vision AI PDF to answer the question:

Context:
{context_text}

User question:
{user_input}

Answer politely and professionally.
"""
    # Use your existing model pipeline
    answer = chatbot(prompt, max_length=200, do_sample=True)[0]["generated_text"]
    return answer

# ========================
# Replace "AI suggestion" in your Streamlit Get Estimate button
# ========================
# suggestion = generate_rag_response(project)
