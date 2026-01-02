import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your kaggle csv here
df = pd.read_csv(r"C:\Users\SREEJA B C\OneDrive\FASTAPI\diabetes_prediction_dataset.csv")

# Simple Pre-processing (Example: encoding gender)
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
# Note: For smoking_history, you should use LabelEncoder or get_dummies
df['smoking_history'] = df['smoking_history'].astype('category').cat.codes

X = df.drop('diabetes', axis=1)
y = df['diabetes']

model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as diabetes_model.pkl")