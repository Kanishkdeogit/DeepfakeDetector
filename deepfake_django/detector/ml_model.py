"""
Model architecture + a process-wide cached loader.

The checkpoint is loaded once per process (not per request) via a module-level
singleton, so repeated requests don't reload ~50MB of weights each time.
"""
import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Path to the trained checkpoint. Override with the DEEPFAKE_MODEL_PATH env var
# if you keep the .pth file somewhere other than the project root.
CHECKPOINT_PATH = os.environ.get(
    "DEEPFAKE_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "custom_cnn_deepfake_detector_final.pth"),
)

_lock = threading.Lock()
_state = {"model": None, "device": None, "val_f1": None, "val_auc": None, "loaded": False, "error": None}


class CustomDeepfakeCNN(nn.Module):
    """Must match the architecture used during training exactly."""

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Same preprocessing as the training notebook's val_transform — no augmentation at inference time.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_model():
    """Load (once) and return (model, device, val_f1, val_auc). Thread-safe."""
    if _state["loaded"]:
        return _state["model"], _state["device"], _state["val_f1"], _state["val_auc"]

    with _lock:
        if _state["loaded"]:
            return _state["model"], _state["device"], _state["val_f1"], _state["val_auc"]

        if not os.path.exists(CHECKPOINT_PATH):
            _state["error"] = f"Checkpoint not found at {CHECKPOINT_PATH}"
            _state["loaded"] = True
            return None, None, None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

        dropout_rate = checkpoint.get("best_hyperparameters", {}).get("dropout_rate", 0.3)
        model = CustomDeepfakeCNN(dropout_rate=dropout_rate).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _state["model"] = model
        _state["device"] = device
        _state["val_f1"] = checkpoint.get("val_macro_f1")
        _state["val_auc"] = checkpoint.get("val_auc")
        _state["loaded"] = True

        return model, device, _state["val_f1"], _state["val_auc"]


def get_load_error():
    return _state["error"]


def predict(image):
    """image: a PIL.Image. Returns (real_prob, fake_prob) as floats."""
    model, device, _, _ = get_model()
    if model is None:
        raise RuntimeError(_state["error"] or "Model not loaded")

    tensor = INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return probs[0].item(), probs[1].item()
