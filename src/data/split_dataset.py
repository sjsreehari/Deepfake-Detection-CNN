# split_dataset.py
# Dataset splitting logic

import os
import logging

def download_dataset():
	"""Download dataset using Kaggle API."""
	"""
	Downloads the dataset from Kaggle using the Kaggle API.
	Requires kaggle.json to be set up in ~/.kaggle or environment variables.
	"""
	import subprocess
	dataset = "deepfake-detection-challenge"
	output_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
	os.makedirs(output_dir, exist_ok=True)
	try:
		subprocess.run([
			"kaggle", "competitions", "download",
			"-c", dataset,
			"-p", output_dir
		], check=True)
		logging.info(f"Downloaded dataset to {output_dir}")
	except Exception as e:
		logging.error(f"Failed to download dataset: {e}")

def copy_images():
	"""Copy images to split folders."""
	"""
	Copies images from the raw dataset to train/val/test split folders.
	"""
	import shutil
	src_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
	dst_dir = os.path.join(os.path.dirname(__file__), '../../data/split')
	os.makedirs(dst_dir, exist_ok=True)
	# Example: Copy all images (stub)
	for fname in os.listdir(src_dir):
		if fname.endswith('.jpg') or fname.endswith('.png'):
			shutil.copy2(os.path.join(src_dir, fname), dst_dir)
	logging.info(f"Copied images to {dst_dir}")

def prepare_splits():
	"""Prepare train/val/test splits."""
	"""
	Splits the dataset into train, val, and test folders.
	"""
	import random
	src_dir = os.path.join(os.path.dirname(__file__), '../../data/split')
	split_dirs = {
		'train': os.path.join(os.path.dirname(__file__), '../../data/train'),
		'val': os.path.join(os.path.dirname(__file__), '../../data/val'),
		'test': os.path.join(os.path.dirname(__file__), '../../data/test'),
	}
	for d in split_dirs.values():
		os.makedirs(d, exist_ok=True)
	images = [f for f in os.listdir(src_dir) if f.endswith('.jpg') or f.endswith('.png')]
	random.shuffle(images)
	n = len(images)
	train_end = int(n * 0.7)
	val_end = int(n * 0.85)
	for i, fname in enumerate(images):
		if i < train_end:
			dst = split_dirs['train']
		elif i < val_end:
			dst = split_dirs['val']
		else:
			dst = split_dirs['test']
		src_path = os.path.join(src_dir, fname)
		dst_path = os.path.join(dst, fname)
		import shutil
		shutil.copy2(src_path, dst_path)
	logging.info(f"Split {n} images into train/val/test folders.")

def main():
	"""Main orchestration for dataset preparation."""
	"""
	Main orchestration for dataset preparation.
	"""
	logging.basicConfig(level=logging.INFO)
	logging.info("Starting dataset preparation...")
	download_dataset()
	copy_images()
	prepare_splits()
	# Dataset summary
	for split in ['train', 'val', 'test']:
		split_dir = os.path.join(os.path.dirname(__file__), f'../../data/{split}')
		count = len([f for f in os.listdir(split_dir) if f.endswith('.jpg') or f.endswith('.png')])
		logging.info(f"{split.capitalize()} images: {count}")


# Entrypoint
if __name__ == "__main__":
	main()
