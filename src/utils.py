# utils.py
# Utility functions (metrics, plots, etc.)

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_confusion_matrix(cm, classes, save_path):
	"""Plot and save confusion matrix."""
	plt.figure(figsize=(5,5))
	plt.imshow(cm, cmap='Blues')
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)
	for i in range(len(classes)):
		for j in range(len(classes)):
			plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()

def plot_training_history(history, save_path):
	"""Plot and save training history (accuracy/loss curves)."""
	acc = history.history.get('accuracy', [])
	val_acc = history.history.get('val_accuracy', [])
	loss = history.history.get('loss', [])
	val_loss = history.history.get('val_loss', [])
	epochs = range(1, len(acc) + 1)

	plt.figure(figsize=(10,4))
	plt.subplot(1,2,1)
	plt.plot(epochs, acc, label='Train Acc')
	plt.plot(epochs, val_acc, label='Val Acc')
	plt.title('Accuracy')
	plt.legend()

	plt.subplot(1,2,2)
	plt.plot(epochs, loss, label='Train Loss')
	plt.plot(epochs, val_loss, label='Val Loss')
	plt.title('Loss')
	plt.legend()

	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()

def save_model(model, save_path):
	"""Save Keras model to file."""
	model.save(save_path)

def load_model(load_path):
	"""Load Keras model from file."""
	return tf.keras.models.load_model(load_path)
