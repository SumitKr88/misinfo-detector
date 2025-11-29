import os
import numpy as np
from PIL import Image

def generate_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Generate a "fake" image (Random Noise)
    # This might trigger the frequency domain check or just look weird to the model
    print("Generating fake image...")
    noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_fake = Image.fromarray(noise)
    img_fake.save(os.path.join(fake_dir, 'noise_fake.jpg'))

    # Generate a "real" image (Gradient or Solid)
    # Note: This lacks EXIF metadata, so the detector might flag it as "Unverified" or "Suspicious"
    # But it serves as a placeholder.
    print("Generating 'real' placeholder image...")
    img_real = Image.new('RGB', (224, 224), color='blue')
    img_real.save(os.path.join(real_dir, 'solid_real.jpg'))

    print("Test data generated.")

if __name__ == "__main__":
    generate_data()
