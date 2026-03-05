"""
Unit and integration tests for the Lightweight CNN Disaster Detection Framework.

Run with::

    pytest tests/ -v

These tests do NOT require a GPU or real disaster images; they use randomly
generated tensors / images so they run quickly in any CI environment.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

# Make sure the src package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import DepthwiseSeparableConv, LightweightDisasterCNN, build_model
from dataset import get_train_transforms, get_eval_transforms, IMAGE_SIZE
from inference import predict_image, load_image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model():
    """Return a freshly initialised model with default settings (CPU)."""
    return LightweightDisasterCNN(num_classes=5)


@pytest.fixture
def small_model():
    """Return a tiny model (width_multiplier=0.25) for fast tests."""
    return LightweightDisasterCNN(num_classes=5, width_multiplier=0.25)


@pytest.fixture
def random_image():
    """Return a random 224×224 RGB PIL image."""
    return Image.fromarray(
        torch.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8).numpy()
    )


@pytest.fixture
def class_names():
    return LightweightDisasterCNN.CLASS_NAMES


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestDepthwiseSeparableConv:
    def test_output_shape_stride1(self):
        block = DepthwiseSeparableConv(32, 64, stride=1)
        x = torch.randn(2, 32, 14, 14)
        out = block(x)
        assert out.shape == (2, 64, 14, 14)

    def test_output_shape_stride2(self):
        block = DepthwiseSeparableConv(32, 64, stride=2)
        x = torch.randn(2, 32, 28, 28)
        out = block(x)
        assert out.shape == (2, 64, 14, 14)

    def test_no_nan_in_output(self):
        block = DepthwiseSeparableConv(16, 32)
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        assert not torch.isnan(out).any()


class TestLightweightDisasterCNN:
    def test_output_shape_default(self, default_model):
        x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = default_model(x)
        assert out.shape == (4, 5)

    def test_output_shape_custom_classes(self):
        model = LightweightDisasterCNN(num_classes=3)
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (2, 3)

    def test_width_multiplier_reduces_params(self, default_model, small_model):
        assert small_model.count_parameters() < default_model.count_parameters()

    def test_parameter_count_positive(self, default_model):
        assert default_model.count_parameters() > 0

    def test_no_nan_in_forward_pass(self, default_model):
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = default_model(x)
        assert not torch.isnan(out).any()

    def test_eval_mode_deterministic(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out1 = small_model(x)
            out2 = small_model(x)
        assert torch.allclose(out1, out2)

    def test_class_names_length(self):
        assert len(LightweightDisasterCNN.CLASS_NAMES) == 5


class TestBuildModel:
    def test_build_model_no_checkpoint(self):
        model = build_model(num_classes=5, device="cpu")
        assert isinstance(model, LightweightDisasterCNN)

    def test_build_model_loads_checkpoint(self, tmp_path, small_model):
        # Save a checkpoint, then reload it
        ckpt = tmp_path / "model.pth"
        torch.save(small_model.state_dict(), ckpt)
        loaded = build_model(
            num_classes=5,
            width_multiplier=0.25,
            checkpoint_path=str(ckpt),
            device="cpu",
        )
        # Verify weights are identical
        for (k1, v1), (k2, v2) in zip(
            small_model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2), f"Mismatch in layer {k1}"

    def test_build_model_loads_checkpoint_dict(self, tmp_path, small_model):
        # Simulate checkpoint format produced by train.py
        ckpt = tmp_path / "best_model.pth"
        torch.save(
            {
                "model_state_dict": small_model.state_dict(),
                "class_names": LightweightDisasterCNN.CLASS_NAMES,
                "epoch": 10,
                "val_acc": 90.0,
            },
            ckpt,
        )
        loaded = build_model(
            num_classes=5,
            width_multiplier=0.25,
            checkpoint_path=str(ckpt),
            device="cpu",
        )
        assert isinstance(loaded, LightweightDisasterCNN)


# ---------------------------------------------------------------------------
# Transform / dataset tests
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_train_transform_output_shape(self, random_image):
        tfm = get_train_transforms()
        tensor = tfm(random_image)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_eval_transform_output_shape(self, random_image):
        tfm = get_eval_transforms()
        tensor = tfm(random_image)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_eval_transform_deterministic(self, random_image):
        tfm = get_eval_transforms()
        t1 = tfm(random_image)
        t2 = tfm(random_image)
        assert torch.allclose(t1, t2)

    def test_train_transform_dtype(self, random_image):
        tfm = get_train_transforms()
        tensor = tfm(random_image)
        assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

class TestPredictImage:
    def test_returns_valid_class(self, small_model, random_image, class_names):
        small_model.eval()
        result = predict_image(small_model, random_image, class_names, torch.device("cpu"))
        assert result["predicted_class"] in class_names

    def test_confidence_in_range(self, small_model, random_image, class_names):
        small_model.eval()
        result = predict_image(small_model, random_image, class_names, torch.device("cpu"))
        assert 0.0 <= result["confidence"] <= 100.0

    def test_probabilities_sum_to_100(self, small_model, random_image, class_names):
        small_model.eval()
        result = predict_image(small_model, random_image, class_names, torch.device("cpu"))
        total = sum(result["probabilities"].values())
        assert abs(total - 100.0) < 0.1, f"Probabilities sum to {total}, expected ~100"

    def test_all_classes_present(self, small_model, random_image, class_names):
        small_model.eval()
        result = predict_image(small_model, random_image, class_names, torch.device("cpu"))
        assert set(result["probabilities"].keys()) == set(class_names)

    def test_predicted_class_has_max_prob(self, small_model, random_image, class_names):
        small_model.eval()
        result = predict_image(small_model, random_image, class_names, torch.device("cpu"))
        max_class = max(result["probabilities"], key=result["probabilities"].get)
        assert result["predicted_class"] == max_class


class TestLoadImage:
    def test_load_valid_image(self, tmp_path, random_image):
        img_path = tmp_path / "test_img.png"
        random_image.save(img_path)
        loaded = load_image(img_path)
        assert loaded.mode == "RGB"
        assert loaded.size == random_image.size

    def test_load_missing_image_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_image(tmp_path / "nonexistent.jpg")
