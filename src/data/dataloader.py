
import tensorflow as tf

def create_dataset(data_dir, batch_size=32, augment=False):
	
	dataset = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		labels='inferred',
		label_mode='categorical',
		batch_size=batch_size,
		image_size=(224, 224),
		shuffle=True
	)

	normalize = tf.keras.layers.Rescaling(1./255)
	dataset = dataset.map(lambda x, y: (normalize(x), y))
	if augment:
		aug_layers = get_augmentation_layers()
		dataset = dataset.map(lambda x, y: (aug_layers(x), y))
	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	return dataset

def get_augmentation_layers():

	return tf.keras.Sequential([
		tf.keras.layers.RandomFlip("horizontal"),
		tf.keras.layers.RandomRotation(0.1),
		tf.keras.layers.RandomZoom(0.1)
	])

def main():

	import os
	data_dir = os.path.join(os.path.dirname(__file__), '../../data/train')
	dataset = create_dataset(data_dir, batch_size=8, augment=True)
	for images, labels in dataset.take(1):
		print(f"Images batch shape: {images.shape}")
		print(f"Labels batch shape: {labels.shape}")

if __name__ == "__main__":
	main()
