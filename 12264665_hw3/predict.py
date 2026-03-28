"""Run inference for HW3 leaf species classification.

Usage example:
  python predict.py --data-dir dataset_hw3 --model-path dataset_hw3/model.pth --output submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMG_SIZE = 224


class LeafDataset(Dataset):
	"""Dataset for reading image paths from csv files used in HW3."""

	def __init__(self, csv_path: Path, images_dir: Path, transform=None):
		self.df = pd.read_csv(csv_path)
		self.images_dir = images_dir
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		rel_path = str(row.iloc[0])
		image_path = self.images_dir / rel_path

		image = Image.open(image_path).convert("RGB")
		if self.transform is not None:
			image = self.transform(image)

		return image, rel_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Predict labels for HW3 test set.")
	parser.add_argument(
		"--data-dir",
		type=str,
		default="dataset_hw3",
		help="Directory containing train.csv, test.csv, and images paths.",
	)
	parser.add_argument(
		"--model-path",
		type=str,
		default=None,
		help="Path to model checkpoint (.pth). If omitted, script tries common defaults.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="submission.csv",
		help="Output CSV path.",
	)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Inference device.",
	)
	return parser.parse_args()


def resolve_data_dir(data_dir_arg: str) -> Path:
	"""Support either dataset_hw3 or dataset_hw3/dataset_hw3 layouts."""
	data_dir = Path(data_dir_arg).expanduser()
	candidates = [data_dir, data_dir / "dataset_hw3"]

	for candidate in candidates:
		if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
			return candidate

	raise FileNotFoundError(
		"Could not find train.csv and test.csv. Tried: "
		+ ", ".join(str(c.resolve()) for c in candidates)
	)


def resolve_model_path(model_path_arg: str | None, data_dir: Path) -> Path:
	if model_path_arg:
		model_path = Path(model_path_arg).expanduser()
		if model_path.exists():
			return model_path
		raise FileNotFoundError(f"Model not found: {model_path}")

	candidates = [
		data_dir / "model.pth",
		Path("model.pth"),
		data_dir.parent / "model.pth",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate

	raise FileNotFoundError(
		"No model checkpoint found. Pass --model-path explicitly. Tried: "
		+ ", ".join(str(c.resolve()) for c in candidates)
	)


def pick_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but not available.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, checkpoint_path: Path, device: torch.device) -> nn.Module:
	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, num_classes)

	state = torch.load(checkpoint_path, map_location=device)
	# Accept either raw state_dict or a wrapped checkpoint.
	if isinstance(state, dict) and "state_dict" in state:
		state = state["state_dict"]

	model.load_state_dict(state)
	model.to(device)
	model.eval()
	return model


def main() -> None:
	args = parse_args()

	data_dir = resolve_data_dir(args.data_dir)
	train_csv_path = data_dir / "train.csv"
	test_csv_path = data_dir / "test.csv"

	train_df = pd.read_csv(train_csv_path)
	class_names = sorted(train_df.iloc[:, 1].unique())
	idx_to_class = {idx: cls_name for idx, cls_name in enumerate(class_names)}

	transform = transforms.Compose(
		[
			transforms.Resize((IMG_SIZE, IMG_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)

	test_dataset = LeafDataset(test_csv_path, data_dir, transform=transform)
	test_loader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
	)

	device = pick_device(args.device)
	model_path = resolve_model_path(args.model_path, data_dir)
	model = build_model(len(class_names), model_path, device)

	predictions = []
	with torch.no_grad():
		for images, image_paths in test_loader:
			images = images.to(device)
			logits = model(images)
			pred_indices = logits.argmax(dim=1).cpu().tolist()

			for image_path, pred_idx in zip(image_paths, pred_indices):
				predictions.append((image_path, idx_to_class[pred_idx]))

	output_df = pd.DataFrame(predictions, columns=["image", "label"])
	output_path = Path(args.output).expanduser()
	output_df.to_csv(output_path, index=False)

	print(f"Data directory: {data_dir.resolve()}")
	print(f"Model path: {model_path.resolve()}")
	print(f"Wrote predictions: {output_path.resolve()} ({len(output_df)} rows)")


if __name__ == "__main__":
	main()
