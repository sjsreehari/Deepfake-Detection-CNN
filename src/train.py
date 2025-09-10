

import os
import tensorflow as tf
from .data.dataloader import create_dataset
from .models.mobilenetv2 import build_mobilenetv2

def main():

	train_dir = os.path.join(os.path.dirname(__file__), '../data/train')
	val_dir = os.path.join(os.path.dirname(__file__), '../data/val')
	train_ds = create_dataset(train_dir, batch_size=32, augment=True)
	val_ds = create_dataset(val_dir, batch_size=32, augment=False)
	print("Loaded train and validation datasets.")

	model, base_model = build_mobilenetv2(input_shape=(224, 224, 3), num_classes=2)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)
	print("Model built and compiled.")


	print("Starting head training phase...")
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=5
	)
	print("Head training complete.")

	# Fine-tuning phase
	print("Starting fine - tuning phase...")
	base_model.trainable = True
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	# Callbacks and checkpoint saving
	checkpoint_dir = os.path.join(os.path.dirname(__file__), '../outputs/checkpoints')
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
		filepath=os.path.join(checkpoint_dir, 'mobilenetv2_best.h5'),
		save_best_only=True,
		monitor='val_accuracy',
		mode='max',
		verbose=1
	)
	earlystop_cb = tf.keras.callbacks.EarlyStopping(
		monitor='val_accuracy',
		patience=3,
		restore_best_weights=True
	)
	csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(checkpoint_dir, 'training_log.csv'))

	# Logging
	import logging
	logging.basicConfig(level=logging.INFO)
	logging.info("Fine-tuning model...")

	history_ft = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=10,
		callbacks=[checkpoint_cb, earlystop_cb, csv_logger]
	)
	logging.info("Fine-tuning complete.")

if __name__ == "__main__":
	main()
