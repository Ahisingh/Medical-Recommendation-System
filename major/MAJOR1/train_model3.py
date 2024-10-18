import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
sym_des = pd.read_csv("./datasets/symtoms_df.csv")
precautions = pd.read_csv("./datasets/precautions_df.csv")
workout = pd.read_csv("./datasets/workout_df.csv")
description = pd.read_csv("./datasets/description.csv")
medications = pd.read_csv('./datasets/medications.csv')
diets = pd.read_csv("./datasets/diets.csv")
training_data = pd.read_csv('./datasets/Training.csv')

# Separate features and target variable
X = training_data.drop('prognosis', axis=1)
y = training_data['prognosis']

# Encode the target variable
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('./randomf_model/rf_model.pkl', 'wb') as file:
    pickle.dump(rf, file)