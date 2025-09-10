import tensorflow as tf

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=2, weights='imagenet', include_top=False):
	"""
	Builds a MobileNetV2 model for deepfake detection using transfer learning.
	Args:
		input_shape (tuple): Shape of input images.
		num_classes (int): Number of output classes.
		weights (str): Pretrained weights to use.
		include_top (bool): Whether to include the top layer.
	Returns:
		model (tf.keras.Model): Compiled model.
		base_model (tf.keras.Model): Base MobileNetV2 model.
	"""
	base_model = tf.keras.applications.MobileNetV2(
		input_shape=input_shape,
		include_top=include_top,
		weights=weights
	)
	base_model.trainable = False
	x = base_model.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(0.3)(x)
	x = tf.keras.layers.Dense(128, activation='relu')(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
	model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
	return model, base_model

