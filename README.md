# Vireum AI RAG Chatbot

A Streamlit-based AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions about your AI projects using DOCX documents as knowledge sources. Supports voice input/output and provides project cost and time estimates.

## Features

- Answer user questions based strictly on uploaded DOCX documents (RAG-based).
-  Voice input and text-to-speech output.
-  Generates estimated cost and duration for AI projects.
    -  Cost estimation is random-based for demo purposes.
-  Caches document embeddings for faster retrieval.
-  Simple and intuitive Streamlit web interface.


## Folder Structure
### Vireum AI Rag Chatbot/
-  ├── vireum_env/           # Python virtual environment
-  ├── data/users.csv        # Saved user inputs (auto-created)
-  ├── docs/                 # DOCX documents for knowledge base
-  ├── embeddings.npy        # Cached embeddings
-   ├── chunks.npy            # Text chunks corresponding to embeddings
-  └── vireum_ai_apuri_rag.py # Main Streamlit + RAG script


## Requirements

*  Python 3.10+

*  Packages (install via requirements.txt or pip):
  
```pip install streamlit sentence-transformers transformers torch pyttsx3 SpeechRecognition python-docx pandas```

## Setup & Running

1. Activate virtual environment
  `vireum_env\Scripts\activate`

optional - Install dependencies:
`pip install -r requirements.txt`


2. Run the Streamlit app
   `python -m streamlit run vireum_ai_apuri_rag.py`

3. Open the URL shown in the console (usually http://localhost:8501) in a browser.

## How to Use

- Fill in your name, email, and project idea.

- Optionally use voice input.

- Click “Get Estimate” to:

  - Retrieve relevant information from DOCX documents.
  - Get AI suggestions in bullet points.
  - Receive estimated cost and project duration.

- Saved entries are stored automatically in data/users.csv.

- Click “Show Saved” to view past submissions.

## Notes

- Adding new documents: Place DOCX files in docs/. Delete embeddings.npy and chunks.npy to regenerate embeddings.

- Model: Uses a small CPU-friendly instruction-following model (google/flan-t5-small) for fast inference.

- Voice support: Requires a working microphone and speakers.

## License

MIT License – feel free to use and modify.
   
### Explanation
- vireum_env/ – virtual environment with all installed packages.
- data/users.csv – stores users’ info, project ideas, and estimates.
- embeddings.npy – caches embeddings from docx so RAG loads faster on restart.
- google_calendar.py – optional file to integrate Google Calendar. Can be removed if we skip calendar functionality.
- vireum_ai_apuri_rag.py – main Streamlit + RAG + voice chatbot script.
- requirements.txt – (optional) useful to reinstall all packages quickly:
