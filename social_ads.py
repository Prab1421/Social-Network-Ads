''' Data Analytics II
1. Implement logistic regression using Python/R to perform classification on 
Social_Network_Ads.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall 
on the given dataset. '''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][1]
    FN = cm[1][0]
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
df = pd.read_csv("Social_Network_Ads.csv")
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
metrics = compute_metrics(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
print(f"True Positive (TP): {metrics['TP']}")
print(f"False Positive (TP): {metrics['FP']}")
print(f"True Negative (TN): {metrics['TN']}")
print(f"False Negative (FN): {metrics['FN']}")
print(f"Accuracy: {accuracy: .2f}")
print(f"Error_Rate: {error_rate: .2f}")
print(f"Precision: {precision: .2f}")
print(f"Recall: {recall: .2f}")

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True,
fmt = "d", cmap = "Blues", xticklabels = ["Not Purchased", "Purchased"],
yticklabels=["Not Purchased", "Purchased"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()