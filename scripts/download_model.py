from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = Path("models/yolov8n.pt")


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading YOLOv8n model to {MODEL_PATH}...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


if __name__ == "__main__":
    download_model()
