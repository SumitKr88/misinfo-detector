# Test Dataset

Place your test images and videos in the corresponding directories:

- `real/`: Place real images and videos here.
- `fake/`: Place AI-generated or deepfake images and videos here.

The evaluation script `evaluate.py` will scan these directories and run the detection model on each file.

## Generating Dummy Data
You can run `python backend/generate_test_data.py` to generate some dummy images for testing the setup.

## Running Evaluation
Run the evaluation script from the project root:
```bash
python backend/evaluate.py
```
Make sure you have installed the local dependencies:
```bash
pip install -r backend/requirements-local.txt
```
