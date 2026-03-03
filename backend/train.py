import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
ratings = pd.read_csv(
    r"C:\Users\Lakshana\Documents\rating_tool\backend\data\ml-100k\ml-100k\u.data",
    sep="\t",
    names=["user_id","movie_id","rating","timestamp"]
)

users = pd.read_csv(
    r"C:\Users\Lakshana\Documents\rating_tool\backend\data\ml-100k\ml-100k\u.user",
    sep="|",
    names=["user_id","age","gender","occupation","zip_code"]
)

df = ratings.merge(users, on="user_id")

# Convert rating → binary
df["liked"] = (df["rating"] >= 4).astype(int)

# Encode categorical
le_gender = LabelEncoder()
le_occ = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["occupation"] = le_occ.fit_transform(df["occupation"])

X = df[["user_id","movie_id","age","gender","occupation"]]
y = df["liked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow
mlflow.set_experiment("Movie-Like-Prediction")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)

# Save local copy for API
joblib.dump(model, "model.joblib")