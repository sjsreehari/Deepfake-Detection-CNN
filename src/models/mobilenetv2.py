import tensorflow as tf

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=2, weights='imagenet', include_top=False):
	"""Builds MobileNetV2 model for deepfake detection."""
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

