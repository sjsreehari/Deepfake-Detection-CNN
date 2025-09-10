


import os
import tensorflow as tf
from .data.dataloader import create_dataset
from .models.mobilenetv2 import build_mobilenetv2
from src.utils_config import load_config

	main()
	# Load config
	config = load_config(os.path.join(os.path.dirname(__file__), '../configs/train.yaml'))

	train_dir = config['dataset']['train_dir']
	val_dir = config['dataset']['val_dir']
	batch_size = config['dataset']['batch_size']
	image_size = tuple(config['dataset']['image_size'])
	augment = config['dataset']['augment']

	train_ds = create_dataset(train_dir, batch_size=batch_size, augment=augment)
	val_ds = create_dataset(val_dir, batch_size=batch_size, augment=False)
	print("Loaded train and validation datasets.")

	backbone = config['model']['backbone']
	weights = config['model']['weights']
	num_classes = config['model']['num_classes']

	# Only MobileNetV2 supported for now
	model, base_model = build_mobilenetv2(input_shape=image_size + (3,), num_classes=num_classes, weights=weights)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate_head']),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)
	print("Model built and compiled.")

	print("Starting head training phase...")
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=config['training']['epochs_head']
	)
	print("Head training complete.")

	# Fine-tuning phase
	print("Starting fine-tuning phase...")
	base_model.trainable = True
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate_finetune']),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	checkpoint_dir = config['checkpoint']['dir']
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
		filepath=os.path.join(checkpoint_dir, config['checkpoint']['best_model']),
		save_best_only=True,
		monitor='val_accuracy',
		mode='max',
		verbose=1
	)
	earlystop_cb = tf.keras.callbacks.EarlyStopping(
		monitor='val_accuracy',
		patience=config['training']['early_stopping_patience'],
		restore_best_weights=True
	)
	csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(checkpoint_dir, config['checkpoint']['log_csv']))

	import logging
	logging.basicConfig(level=getattr(logging, config['logging']['level']))
	logging.info("Fine-tuning model...")

	history_ft = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=config['training']['epochs_finetune'],
		callbacks=[checkpoint_cb, earlystop_cb, csv_logger]
	)
	logging.info("Fine-tuning complete.")

if __name__ == "__main__":
	main()
