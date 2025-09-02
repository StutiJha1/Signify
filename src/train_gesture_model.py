import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Path to your extracted CSV folder
dataset_folder = "../extracted_data"

# Load all CSV files
all_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]

data = []
labels = []

# Read data from each CSV file
for file in all_files:
    df = pd.read_csv(file)
    letter = os.path.splitext(os.path.basename(file))[0]  # get letter from filename
    for _, row in df.iterrows():
        data.append(row.values)
        labels.append(letter)

# Convert to numpy arrays
X = pd.DataFrame(data).values
y = pd.Series(labels).values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Test accuracy
y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(clf, "hand_gesture_model.pkl")
print("Model saved as hand_gesture_model.pkl")
