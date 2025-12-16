from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

actual = [1, 0, 1, 1, 0, 1]
predicted = [1, 1, 1, 0, 0, 0]

cm = confusion_matrix(actual, predicted)
tn, fp, fn, tp = cm.ravel()

print("Confusion matrix:")
print(cm) 
print(f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}")

accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)
f1 = f1_score(actual, predicted)

print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1-score  = {f1:.4f}")