# 🎬 Movie Like Prediction API

An end-to-end Machine Learning project that predicts whether a user will like a movie.

Built using:
- FastAPI
- Scikit-learn
- MLflow (Experiment Tracking)
- Random Forest Classifier
- MovieLens 100K Dataset

---

## 🚀 Project Overview

This project:

1. Trains a machine learning model on the MovieLens 100K dataset
2. Tracks experiments using MLflow
3. Saves trained model artifact
4. Deploys prediction API using FastAPI
5. Returns prediction with probability score

---

## 📊 Dataset

MovieLens 100K Dataset  
100,000 ratings from 943 users on 1,682 movies.

Target:
- rating >= 4 → 1 (Liked)
- rating < 4 → 0 (Not Liked)

---

## 🧠 Model

- Algorithm: RandomForestClassifier
- Features:
  - user_id
  - movie_id
  - age
  - gender (encoded)
  - occupation (encoded)

Evaluation Metric:
- Accuracy

---

## 🛠 Installation

### 1️⃣ Clone the repo

```bash
git clone <your-repo-url>
cd backend
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python train.py
```

This will:
- Train model
- Log experiment in MLflow
- Save `model.joblib`

To view experiments:

```bash
mlflow ui
```

Open:
```
http://127.0.0.1:5000
```

---

## 🚀 Run API Locally

```bash
uvicorn app:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## 📌 API Usage

### POST `/predict`

Example Request:

```json
{
  "user_id": 10,
  "movie_id": 50,
  "age": 25,
  "gender": 1,
  "occupation": 3
}
```

Example Response:

```json
{
  "will_like": true,
  "confidence": 0.87
}
```

---

## ☁️ Deployment

This project can be deployed on:

- Render (recommended free tier)
- Railway
- Fly.io

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

## 📈 Future Improvements

- Hyperparameter tuning
- Add genre features
- Model registry with MLflow
- Drift detection
- CI/CD pipeline

---

## 🏆 Skills Demonstrated

- End-to-end ML pipeline
- Feature engineering
- Experiment tracking (MLflow)
- API development (FastAPI)
- Model deployment
- MLOps fundamentals

---

## 👨‍💻 Author

Lakshana Gowrishankar
