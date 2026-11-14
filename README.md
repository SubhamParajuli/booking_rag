# Booking RAG

A simple document-based booking assistant with a FastAPI backend and Streamlit frontend.

## Requirements

- Python 3.11+
- Docker (for Redis)
- A [Groq API key](https://console.groq.com/keys)

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

4. Start Redis:

   ```powershell
   docker compose up -d redis
   ```

## Run the project

Open two terminals, activate the virtual environment in both, and run:

**Terminal 1 - API**

```powershell
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend**

```powershell
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL shown in the terminal. The API documentation is available at
`http://127.0.0.1:8000/docs`.

## Stop Redis

```powershell
docker compose down
```
