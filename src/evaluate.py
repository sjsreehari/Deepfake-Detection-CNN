
# evaluate.py
# Evaluation and metrics script for Deepfake Detector
#
# Loads the trained model and test dataset, computes metrics, and saves plots.

import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def main():
	"""
	Evaluate the trained deepfake detection model on the test dataset.
	Computes accuracy, classification report, ROC AUC, and saves confusion matrix and ROC curve plots.
	"""
	# Load test dataset
	test_dir = os.path.join(os.path.dirname(__file__), '../data/test')
	batch_size = 32
	from src.data.dataloader import create_dataset
	test_ds = create_dataset(test_dir, batch_size=batch_size, augment=False)

	# Load trained model
	checkpoint_path = os.path.join(os.path.dirname(__file__), '../outputs/checkpoints/mobilenetv2_best.h5')
	model = tf.keras.models.load_model(checkpoint_path)
	print("Model loaded for evaluation.")

	# Prediction loop
	y_true = []
	y_pred = []
	y_prob = []
	for images, labels in test_ds:
		preds = model.predict(images)
		y_true.extend(np.argmax(labels, axis=1))
		y_pred.extend(np.argmax(preds, axis=1))
		y_prob.extend(preds[:, 1])  # Assuming class 1 is 'fake'

	# Compute metrics
	acc = accuracy_score(y_true, y_pred)
	report = classification_report(y_true, y_pred)
	auc = roc_auc_score(y_true, y_prob)
	print(f"Test Accuracy: {acc:.4f}")
	print("Classification Report:\n", report)

	# Plot confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	plt.figure(figsize=(5,5))
	plt.imshow(cm, cmap='Blues')
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.colorbar()
	plt.savefig(os.path.join(os.path.dirname(__file__), '../outputs/plots/confusion_matrix.png'))
	plt.close()

	# Plot ROC curve
	fpr, tpr, _ = roc_curve(y_true, y_prob)
	plt.figure()
	plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc='lower right')
	plt.savefig(os.path.join(os.path.dirname(__file__), '../outputs/plots/roc_curve.png'))
	plt.close()
	print("ROC curve plot saved.")

if __name__ == "__main__":
	main()