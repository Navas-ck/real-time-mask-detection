# Face Mask and PPE Detection

This project provides real-time detection of face masks and personal protective equipment (PPE) using computer vision and deep learning models. It includes scripts for detecting masks on faces and PPE on upper bodies via webcam.

## Features

- **Face Mask Detection**: Detects whether individuals are wearing face masks in real-time using a webcam.
- **PPE Detection**: Detects whether individuals are wearing PPE (e.g., vests, helmets) on their upper bodies.
- **Model Training**: Includes a script to train a PPE detection model using MobileNetV2.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- (Optional) PyTorch and TorchVision if using additional features

Install the dependencies using:

```bash
pip install tensorflow opencv-python numpy
```

Or if using the provided `req.txt` (note: it includes PyTorch, which may be for other parts of the project):

```bash
pip install -r req.txt
```

## Files Overview

- `detection.py`: Real-time face mask detection using webcam.
- `ppe.py`: Real-time PPE detection on upper bodies using webcam.
- `ppe_detection.py`: Script to train a PPE detection model.
- `mask_detector_strong.h5`: Pre-trained model for face mask detection.
- `ppe_detector.h5`: Pre-trained model for PPE detection.
- `model.py`: Additional model-related code (if any).
- `maskdetecter.py`: Alternative or additional mask detection script.
- `req.txt`: Requirements file.

## Usage

### Face Mask Detection

Run the face mask detection script:

```bash
python detection.py
```

This will open your webcam and detect faces, classifying them as "with_mask" or "without_mask". Detected faces are highlighted with bounding boxes and labels.

### PPE Detection

Run the PPE detection script:

```bash
python ppe.py
```

This will detect upper bodies via webcam and classify them as "WITH PPE" or "WITHOUT PPE".

### Training PPE Model

To train the PPE detection model (requires dataset):

```bash
python ppe_detection.py
```

Ensure you have the dataset in `dataset_ppe/train`, `dataset_ppe/val`, and `dataset_ppe/test` directories.

## Notes

- The pre-trained models (`mask_detector_strong.h5` and `ppe_detector.h5`) are included. If you need to retrain, use the respective training scripts.
- Ensure your webcam is accessible and not in use by other applications.
- For best results, use good lighting and position the camera appropriately.

## Troubleshooting

- If the model fails to load, ensure the `.h5` files are in the same directory as the scripts.
- If OpenCV cascades fail, ensure OpenCV is properly installed.
- For training, make sure TensorFlow and the dataset are correctly set up.

## dataset you can download from kaggle or my data link is below
https://drive.google.com/drive/folders/1nKZQM5lJS6RAccfiHKeNCXm853JojlQK?usp=drive_link

## captured images
https://drive.google.com/drive/folders/1Hv9H1L_H-5z3hv1KQMA-udli-FYhGHmt?usp=drive_link


## capture examples
<img width="628" height="472" alt="image" src="https://github.com/user-attachments/assets/deceb7a8-7893-4960-a1ac-5b6529c3e10a" />
<img width="607" height="466" alt="image" src="https://github.com/user-attachments/assets/1d79ceae-9b47-4030-abf2-dd00b0dc42c8" />
<img width="633" height="472" alt="image" src="https://github.com/user-attachments/assets/82324369-c7c4-49f1-acb6-299a2d15e083" />
<img width="627" height="467" alt="image" src="https://github.com/user-attachments/assets/645bcb0c-86f5-4735-8eae-87df52c0a2ca" />

## Contributing

Feel free to contribute by submitting issues or pull requests.
