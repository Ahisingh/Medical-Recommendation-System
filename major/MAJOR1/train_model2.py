import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to numpy arrays
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

# Define the linear kernel function
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Implement the SVM optimization using gradient descent
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.1, n_iters=2000):
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

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {self._calculate_loss(X, y_)}")
                print(f"Iteration {i}: Weights = {self.w}, Bias = {self.b}")

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def _calculate_loss(self, X, y):
        distances = 1 - y * (np.dot(X, self.w) - self.b)
        distances[distances < 0] = 0  # max(0, distance)
        hinge_loss = self.lambda_param * (np.sum(distances) / len(y))
        return 0.5 * np.dot(self.w, self.w) + hinge_loss

# Initialize and train the model
svm = SVM(learning_rate=0.01, lambda_param=0.1, n_iters=2000)
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('./model2/svm_manual1.pkl', 'wb') as file:
    pickle.dump(svm, file)