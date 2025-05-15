# 🎬 Movie Recommender System

This is a modular, production-ready movie recommendation system built using Python, FastAPI, and modern ML tooling.

### 📦 Features
- Content-based filtering with TF-IDF
- Real-time metadata from TMDb API
- Scalable FastAPI backend
- Ready for extension into hybrid or collaborative filtering

### 📂 Folder Structure
- `backend/` – REST API server (FastAPI)
- `recommender/` – Core recommendation engine logic
- `data/` – Scripts + raw/processed datasets
- `tests/` – Unit + integration tests

---

## 🚀 To Run:
```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


---

## 📄 1.4 `requirements.txt`

Start lean with essential packages:

Core
pandas
numpy
scikit-learn
requests

API
fastapi
uvicorn

NLP
nltk
scipy

Dev
python-dotenv

streamlit
imdbpy
sentence-transformers
sqlalchemy
lightfm

bash
Copy
Edit
