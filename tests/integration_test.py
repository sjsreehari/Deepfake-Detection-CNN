import os
import unittest
import tensorflow as tf
from src.data.dataloader import create_dataset
from src.models.mobilenetv2 import build_mobilenetv2
from src.utils import save_model, load_model

class TestDeepfakePipeline(unittest.TestCase):
    def test_full_pipeline(self):
        # Dataset
        train_dir = os.path.join(os.path.dirname(__file__), '../data/train')
        val_dir = os.path.join(os.path.dirname(__file__), '../data/val')
        train_ds = create_dataset(train_dir, batch_size=8, augment=True)
        val_ds = create_dataset(val_dir, batch_size=8, augment=False)
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(val_ds)

        # Model
        model, base_model = build_mobilenetv2(input_shape=(224, 224, 3), num_classes=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.assertIsNotNone(model)

        # Training
        history = model.fit(train_ds, validation_data=val_ds, epochs=1)
        self.assertIsNotNone(history)

        # Save/Load
        save_path = os.path.join(os.path.dirname(__file__), '../outputs/checkpoints/test_model.h5')
        save_model(model, save_path)
        loaded_model = load_model(save_path)
        self.assertIsNotNone(loaded_model)

        # Evaluation
        results = loaded_model.evaluate(val_ds)
        self.assertIsNotNone(results)

if __name__ == "__main__":
    unittest.main()
