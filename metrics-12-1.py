from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

actual = [1, 0, 1, 0, 1]
predicted = [1, 1, 1, 0, 0]

cm = confusion_matrix(actual, predicted)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(actual, predicted)
print("Accuracy:", accuracy)

precision = precision_score(actual, predicted)
print("Precision:", precision)

recall = recall_score(actual, predicted)
print("Recall:", recall)

f1 = f1_score(actual, predicted)
print("F1-Score:", f1)
