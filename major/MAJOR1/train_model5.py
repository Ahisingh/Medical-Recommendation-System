import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from collections import Counter

# Load datasets
print("Loading datasets...")
sym_des = pd.read_csv("./datasets/symtoms_df.csv")
precautions = pd.read_csv("./datasets/precautions_df.csv")
workout = pd.read_csv("./datasets/workout_df.csv")
description = pd.read_csv("./datasets/description.csv")
medications = pd.read_csv('./datasets/medications.csv')
diets = pd.read_csv("./datasets/diets.csv")
training_data = pd.read_csv('./datasets/Training.csv')

# Separate features and target variable
print("Separating features and target variable...")
X = training_data.drop('prognosis', axis=1)
y = training_data['prognosis'].values
# Check the distribution of diseases in the training dataset
disease_counts = pd.Series(y).value_counts()
print("Disease distribution in training data:")
print(disease_counts)
# Encode the target variable
print("Encoding the target variable...")
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# Normalize the data
print("Normalizing the data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None  # Initialize feature importances

    def fit(self, X, y):
        print("Fitting a decision tree...")
        self.tree = self._grow_tree(X, y)
        self.feature_importances_ = self._calculate_feature_importances()  # Calculate feature importances
        print("Decision tree fitted.")

    def _grow_tree(self, X, y, depth=0):
        print(f"Depth: {depth}, X shape: {X.shape}, y shape: {y.shape}")  # Debugging line
    # Check if we should stop growing the tree
        if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=self._most_common_label(y))

    # Find the best split
        num_features = X.shape[1]
        best_feature, best_threshold = self._best_split(X, y, num_features)

    # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

    # Ensure that the indices are valid
        print(f"Left indices count: {np.sum(left_indices)}, Right indices count: {np.sum(right_indices)}")  # Debugging line

    # Check if there are samples in both splits
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return Node(value=self._most_common_label(y))  # Return a leaf node if no valid split

        left_node = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _best_split(self, X, y, num_features):
        # Implement logic to find the best feature and threshold for splitting
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _calculate_feature_importances(self):
        importances = np.zeros(X.shape[1])  # Initialize importances
        self._traverse_tree(self.tree, importances)  # Traverse the tree to calculate importances
        return importances

    def predict(self, X):
        predictions = [self._predict(sample) for sample in X]
        return np.array(predictions)

    def _predict(self, sample):
        node = self.tree
        while not node.is_leaf_node():
            if sample[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _traverse_tree(self, node, importances):
        if node.is_leaf_node():
            return
    # Increment the importance of the feature used for the split
        importances[node.feature] += 1  # You can adjust this based on the split quality
        self._traverse_tree(node.left, importances)
        self._traverse_tree(node.right, importances)


    def _most_common_label(self, y):
        # Return the most common label in y
        return Counter(y).most_common(1)[0][0]
    
    def _information_gain(self, X, y, feature, threshold):
    # Calculate the parent impurity
        parent_impurity = self._gini_impurity(y)

    # Split the dataset
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

    # Calculate the weighted impurity of the children
        n = len(y)
        n_left = np.sum(left_indices)
        n_right = np.sum(right_indices)

        if n_left == 0 or n_right == 0:
            return 0  # No split

        left_impurity = self._gini_impurity(y[left_indices])
        right_impurity = self._gini_impurity(y[right_indices])

    # Weighted impurity
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

    # Information gain
        return parent_impurity - child_impurity

    def _gini_impurity(self, y):
    # Calculate Gini impurity
        m = len(y)
        if m == 0:
            return 0
        p = np.bincount(y) / m
        return 1 - np.sum(p ** 2)

    # ... (rest of your DecisionTree methods remain unchanged)

    # ... (rest of your DecisionTree methods remain unchanged)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
          # Initialize feature importances
    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices].values, y[indices]  # Use .iloc for DataFrame indexing

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.feature_importances_ = np.zeros(X.shape[1])  # Initialize feature importances
        for i in range(self.n_estimators):
            print(f"Training tree {i + 1}/{self.n_estimators}...")
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_  # Assuming tree has this attribute
        self.feature_importances_ /= self.n_estimators  # Average the importances
        print("Random Forest fitted.")

    

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    # Debugging: Print predictions from each tree
        

# Initialize the Random Forest model
print("Initializing the Random Forest model...")
rf = RandomForest(n_estimators=200, max_depth=10, random_state=42)


# Train the model

# Cross-validation scores
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X_train, y_train, cv=5)  # Use training data
print("Cross-validation scores:", scores)
# Train the model
# After the model training code
# Train the model
print("Training the model...")
rf.fit(X_train, y_train)

# After training the model
importances = rf.feature_importances_
print("Feature importances:", importances)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {X.columns[indices[f]]} ({importances[indices[f]]})")

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = rf.predict(X_test)

# Add the following lines after line 170
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:")
print(report)


# Calculate accuracy
print("Calculating accuracy...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create a mapping of symptoms to indices
symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
print("Available symptoms:", list(symptoms_dict.keys()))
# Function to get user input and predict disease
# Function to get user input and predict disease
def predict_disease():
    symptoms_input = input("Enter your symptoms (comma-separated): ")
    user_symptoms = [s.strip() for s in symptoms_input.split(',')]
    
    # Create an input vector based on the symptoms
    input_vector = np.zeros(X.shape[1])  # Ensure this matches the number of features  # Ensure this matches the number of features
    for symptom in user_symptoms:
        if symptom in symptoms_dict:
            index = symptoms_dict[symptom]
            input_vector[index] = 1  # Set the index corresponding to the symptom to 1
        else:
            print(f"Warning: '{symptom}' is not recognized as a valid symptom.")
    
    # Make prediction
    predicted_disease_index = rf.predict([input_vector])[0]
    predicted_disease = le.inverse_transform([predicted_disease_index])[0]
    print(f"Predicted Disease: {predicted_disease}")
if __name__ == "__main__":
    predict_disease()

# Calculate accuracy
#print("Calculating accuracy...")
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
#print("Saving the model...")
#with open('./randomf_model/rf_model_manual1.pkl', 'wb') as file:
#    pickle.dump(rf, file)

#print("Done.")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X, Y, cv=5)
print("Cross-validation scores:", scores)