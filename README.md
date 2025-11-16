Explanation

vireum_env/ – virtual environment with all installed packages.

data/users.csv – stores users’ info, project ideas, and estimates.

embeddings.npy – caches embeddings from docx so RAG loads faster on restart.

google_calendar.py – optional file to integrate Google Calendar. Can be removed if we skip calendar functionality.

vireum_ai_apuri_rag.py – main Streamlit + RAG + voice chatbot script.

requirements.txt – (optional) useful to reinstall all packages quickly:
