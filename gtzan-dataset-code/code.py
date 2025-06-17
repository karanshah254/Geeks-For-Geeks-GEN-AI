# 📌 Step 1: Install required packages (if not already installed)
# !pip install kagglehub librosa scikit-learn numpy matplotlib

# 📌 Step 2: Import all required libraries
import kagglehub
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# 📥 Step 3: Download dataset using kagglehub
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print("Path to dataset files:", path)


# 📊 Step 4: Define function to extract MFCC audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)


# 📁 Step 5: Load and preprocess the dataset
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
features = []
labels = []

base_path = os.path.join(path, "Data/genres_original")  # Correct path to genre folders

for genre in genres:
    genre_dir = os.path.join(base_path, genre)
    for file in os.listdir(genre_dir):
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(genre_dir, file)
                data = extract_features(file_path)
                features.append(data)
                labels.append(genre)
            except Exception as e:
                print(f"Error processing {file}: {e}")


# 📊 Step 6: Convert to NumPy arrays and encode labels
X = np.array(features)
y = np.array(labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)


# 📊 Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# 🤖 Step 8: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


# 📈 Step 9: Evaluate Model
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# 📷 Step 10: Visualize sample audio and MFCC (Optional)
sample_file = os.path.join(base_path, 'blues', 'blues.00000.wav')  # Change genre & file name as needed
y, sr = librosa.load(sample_file)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.show()
