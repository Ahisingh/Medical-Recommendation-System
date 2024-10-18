import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
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

# Convert data to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Define the linear kernel function
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Implement the SVM optimization using gradient descent
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Initialize and train the model
svm = SVM()
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('./model2/svm_manual.pkl', 'wb') as file:
    pickle.dump(svm, file)