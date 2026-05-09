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

## License

[Add your license here, e.g., MIT]

## Contributing

Feel free to contribute by submitting issues or pull requests.