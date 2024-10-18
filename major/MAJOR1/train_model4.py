import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

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

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

# Initialize the Random Forest model
rf = RandomForest(n_estimators=100, max_depth=None, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('./randomf_model/rf_model_manual.pkl', 'wb') as file:
    pickle.dump(rf, file)