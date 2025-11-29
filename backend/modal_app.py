import os
import modal
try:
    from detector_logic import DeepfakeDetectorLogic
except ImportError:
    from backend.detector_logic import DeepfakeDetectorLogic

# Define the local path to the logic file
# We assume this file is in the same directory as modal_app.py
local_logic_path = os.path.join(os.path.dirname(__file__), "detector_logic.py")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "opencv-python-headless",
        "numpy",
        "requests",
        "timm",
        "scipy"
    )
    .add_local_file(local_logic_path, remote_path="/root/detector_logic.py")
)

app = modal.App("deepfake-detector-mvp")

@app.cls(image=image, gpu="T4", timeout=600)
class DeepfakeDetector(DeepfakeDetectorLogic):

    @modal.enter()
    def setup(self):
        self.load_models()

    @modal.method()
    def analyze_media(self, file_url: str, file_type: str):
        import requests
        import numpy as np

        # A. Download
        local_filename = "/tmp/input_media"
        try:
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            return {"status": "error", "message": f"Download failed: {str(e)}"}

        return self.analyze_local_file(local_filename, file_type)