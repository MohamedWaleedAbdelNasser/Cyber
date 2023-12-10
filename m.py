import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'D:\\machine_learning\\UNSW-NB15_1.csv'

# Specify dtype for columns that are causing issues or set low_memory=False to silence the warning
data = pd.read_csv(file_path, low_memory=False)

# Remove columns with IP addresses or other non-numeric data that won't be used
data = data.drop(columns=data.columns[[1, 3, 47]])

# Convert all remaining object columns to numeric and fill NaNs with 0
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')
data.fillna(0, inplace=True)

# Shuffle the data to avoid any order bias
data = shuffle(data, random_state=42)

# Separate the features and the target labels
X = data.iloc[:, :-2]  # Features: all columns except the last two
y = data.iloc[:, -2]   # Labels: the second to last column (attack types)

# Encode the attack names as integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize the XGBoost Classifier
clf = XGBClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train_encoded)

# Predict the labels of the test set
y_pred_encoded = clf.predict(X_test)

# Reverse the encoded labels back to original attack names for evaluation
y_test = encoder.inverse_transform(y_test_encoded)
y_pred = encoder.inverse_transform(y_pred_encoded)

# Calculate metrics for each attack
attack_types = encoder.classes_
metrics = {attack: {'Precision': 0, 'Recall': 0, 'F1-Measure': 0, 'Accuracy': 0} for attack in attack_types}

for attack in attack_types:
    # Filter the relevant true positives, false positives, false negatives
    true_positive = ((y_test == attack) & (y_pred == attack)).sum()
    false_positive = ((y_test != attack) & (y_pred == attack)).sum()
    false_negative = ((y_test == attack) & (y_pred != attack)).sum()
    true_negative = ((y_test != attack) & (y_pred != attack)).sum()
    
    # Calculate the metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(y_test)
    
    # Store the metrics
    metrics[attack]['Precision'] = precision
    metrics[attack]['Recall'] = recall
    metrics[attack]['F1-Measure'] = f1
    metrics[attack]['Accuracy'] = accuracy

# Print out the metrics for each attack
for attack, scores in metrics.items():
    print(f"Attack Type: {attack}")  # This will print the actual attack name
    print(f"Precision: {scores['Precision']:.2f}")
    print(f"Recall: {scores['Recall']:.2f}")
    print(f"F1-Measure: {scores['F1-Measure']:.2f}")
    print(f"Accuracy: {scores['Accuracy']:.2f}\n")
