# Multi-Location Review Management System

AI-powered customer review analysis and response system with a FastAPI backend and Streamlit frontend. Supports sentiment analysis, topic extraction, analytics, search, and AI-generated reply suggestions.

---

## Features

- Ingest customer reviews from multiple locations
- Rating-based and text-based sentiment analysis
- Topic extraction (service, food, cleanliness, price, ambiance)
- Analytics by location, sentiment, rating, and topic
- Search similar reviews using TF-IDF and cosine similarity
- AI-powered reply suggestion (OpenAI or local template-based)
- Streamlit frontend for interactive UI

---

## Tech Stack

- **Backend**: FastAPI, SQLite, scikit-learn, Hugging Face Transformers (optional)
- **Frontend**: Streamlit
- **AI**: OpenAI API (optional) or local fallback using Transformers and templates
- **Deployment**: Render (backend + Streamlit frontend separately)

---

## Requirements

- Python 3.11 (required for Torch & Transformers compatibility)
- Packages in `requirements.txt`:

```txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
scikit-learn==1.5.2
transformers==4.35.2
torch==2.1.1
sentencepiece==0.1.99
streamlit==1.27.0



Setup

Clone the repository

git clone <your-repo-url>
cd <repo-folder>


Set Python version (Render)

Create a runtime.txt:

python-3.11.9


Create virtual environment

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows


Install dependencies

pip install --upgrade pip
pip install -r requirements.txt


Environment variables

Create a .env file:

API_KEY=demo-key-12345
OPENAI_API_KEY=<your-openai-key>  # optional
DB_PATH=reviews.db

Run Locally

Backend (FastAPI)

uvicorn main:app --reload --host 0.0.0.0 --port 8000


Open API docs: http://localhost:8000/docs

Frontend (Streamlit)

streamlit run frontend.py --server.port 8501


Open Streamlit app: http://localhost:8501

Deployment on Render

Backend (FastAPI)

Create a new Web Service

Connect GitHub repo

Set runtime.txt to Python 3.11

Build command: pip install -r requirements.txt

Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

Frontend (Streamlit)

Create a new Web Service

Connect GitHub repo

Set runtime.txt to Python 3.11

Build command: pip install -r requirements.txt

Start command: streamlit run frontend.py --server.port $PORT

Make sure to deploy backend and frontend as separate services if they share the same repository.

API Endpoints

POST /ingest - Ingest multiple reviews

GET /reviews - Get filtered reviews

GET /reviews/{id} - Get a single review

DELETE /reviews/{id} - Delete review

POST /reviews/{id}/suggest-reply - Generate AI reply

GET /analytics - Analytics summary

GET /analytics/by-location - Analytics by location

GET /search?q=<query> - Search similar reviews

All endpoints require x-api-key header.

Notes

Use Python 3.11 for ML libraries compatibility.

If OpenAI API key is not provided, reply generation will use local templates.

Streamlit frontend communicates with FastAPI backend via API.
