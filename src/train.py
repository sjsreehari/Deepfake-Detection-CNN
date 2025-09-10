

import os
import tensorflow as tf
from src.data.dataloader import create_dataset
from src.models.mobilenetv2 import build_mobilenetv2

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

if __name__ == "__main__":
	main()
