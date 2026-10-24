import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 1. Portable project paths
PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "audio_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "model_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "embeddings_cache.pkl"
MODEL_PATH = OUTPUT_DIR / "scream_classifier.pkl"
FAILED_PATH = OUTPUT_DIR / "failed_files.csv"
# 2. Load dataset
print("Loading dataset:", CSV_PATH)
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset CSV not found: {CSV_PATH}"
    )
df = pd.read_csv(CSV_PATH)
required_columns = [
    "file_name", "file_path", "folder_name", "label"
]
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing CSV column: {column}")
print("Total CSV records:", len(df))
print("\nOriginal labels:")
print(df["label"].value_counts(dropna=False))
# 3. Prepare labels and portable audio paths
LABEL_MAPPING = {
    "scream": "scream",
    "not_scream": "not_scream",
    "nonverbal_vocalization_scream": "scream"
}
df["target"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(LABEL_MAPPING)
)
# Keep only rows with usable training labels
df = df.dropna(subset=["target"]).copy()
def resolve_audio_path(row):
    """
    First try the path saved in the CSV.
    If it belongs to another computer, use the
    project folder + folder_name + file_name.
    """
    original_path = str(row["file_path"])
    if os.path.isfile(original_path):
        return Path(original_path).resolve()
    relative_path = (
        PROJECT_DIR
        / str(row["folder_name"])
        / str(row["file_name"])
    )
    if relative_path.is_file():
        return relative_path.resolve()
    return None
print("\nFinding audio files...")
df["resolved_path"] = [
    resolve_audio_path(row)
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Checking paths"
    )
]
missing_count = int(df["resolved_path"].isna().sum())
df = df.dropna(subset=["resolved_path"]).copy()
df["resolved_path"] = df["resolved_path"].astype(str)
df = df.drop_duplicates(subset=["resolved_path"])
print("Missing audio files:", missing_count)
print("Usable labeled files:", len(df))
print("\nTraining classes:")
print(df["target"].value_counts())
if df["target"].nunique() != 2:
    raise ValueError(
        "Both scream and not_scream classes are required."
    )
# 4. Load YAMNet
print("\nLoading YAMNet...")
yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)
print("YAMNet loaded successfully!")
# 5. Extract audio embeddings
def extract_embedding(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=16000,
        mono=True
    )
    if len(audio) == 0:
        raise ValueError("Empty audio file")
    audio = audio.astype(np.float32)
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(
        embeddings, axis=0
    ).numpy()
    return embedding.astype(np.float32)
# 6. Load or create feature cache
if CACHE_PATH.exists():
    cache = joblib.load(CACHE_PATH)
    print("\nExisting cache loaded:", len(cache))
else:
    cache = {}
X = []
y = []
failed_files = []
print("\nExtracting/loading features...")
for row in tqdm(
    df.itertuples(index=False),
    total=len(df),
    desc="Preparing features"
):
    path = row.resolved_path
    label = row.target
    # Relative cache key works across computers
    cache_key = os.path.relpath(path, PROJECT_DIR)
    try:
        if cache_key in cache:
            embedding = cache[cache_key]
        elif path in cache:
            # Compatibility with an older cache
            # that used absolute paths as keys.
            embedding = cache[path]
        else:
            embedding = extract_embedding(path)
        embedding = np.asarray(
            embedding, dtype=np.float32
        ).reshape(-1)
        if embedding.shape[0] != 1024:
            raise ValueError("Expected 1024 features")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Invalid feature values")
        cache[cache_key] = embedding
        X.append(embedding)
        y.append(label)
    except Exception as error:
        failed_files.append({
            "file_path": path,
            "label": label,
            "error": str(error)
        })
    # Save progress every 100 processed rows
    if len(X) > 0 and len(X) % 100 == 0:
        joblib.dump(cache, CACHE_PATH)
joblib.dump(cache, CACHE_PATH)
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
print("\nFeature preparation complete.")
print("Successful files:", len(X))
print("Failed files:", len(failed_files))
print("Feature matrix:", X.shape)
if failed_files:
    pd.DataFrame(failed_files).to_csv(
        FAILED_PATH,
        index=False,
        encoding="utf-8-sig"
    )
    print("Failed-file report:", FAILED_PATH)
if len(X) < 10 or len(np.unique(y)) != 2:
    raise ValueError(
        "Not enough valid features or one class is missing."
    )
# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# 8. Train classifier
print("\nTraining Logistic Regression...")
classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
classifier.fit(X_train, y_train)
print("Training completed!")
# 9. Evaluate model
y_pred = classifier.predict(X_test)
print(
    "\nAccuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["not_scream", "scream"],
        zero_division=0
    )
)
print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["not_scream", "scream"]
    )
)
# 10. Save trained model
joblib.dump(classifier, MODEL_PATH)
print("\nModel saved:", MODEL_PATH)
# 11. Test a sample audio file
sample_path = df.iloc[0]["resolved_path"]
sample_embedding = extract_embedding(sample_path)
sample_embedding = sample_embedding.reshape(1, -1)
prediction = classifier.predict(sample_embedding)[0]
probabilities = classifier.predict_proba(sample_embedding)[0]
confidence = float(np.max(probabilities)) * 100
print("\nSample audio test")
print("File:", os.path.basename(sample_path))
print("Predicted class:", prediction)
print("Confidence:", round(confidence, 2), "%")
print("\nAll tasks completed!")
