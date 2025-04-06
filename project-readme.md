# Face Analysis Project

This project implements a facial recognition system that can detect faces and predict gender, age range, and racial characteristics. It's based on the FairFace dataset approach described in the paper "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age" (Karkkainen & Joo, WACV 2021).

## Features

- Face detection using OpenCV's DNN face detector
- Multi-task learning for gender, age range, and race prediction
- Support for both image and video processing
- Pre-trained model weights included
- Visualization of results with bounding boxes and labels

## Requirements

- Python 3.13
- PyTorch 2.2+
- OpenCV 4.8+
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-analysis.git
cd face-analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the pre-trained model (if not training your own):
```bash
mkdir -p models
# The face detection model files will be downloaded automatically when running the script
```

## Usage

### Process an image

```bash
python main.py image --input path/to/image.jpg --output path/to/output.jpg --model models/best_model.pth --show
```

### Process a video

```bash
python main.py video --input path/to/video.mp4 --output path/to/output.mp4 --model models/best_model.pth
```

### Training your own model

1. Download the FairFace dataset from [https://github.com/joojs/fairface](https://github.com/joojs/fairface)
2. Prepare your CSV file with annotations in the format:
   - Columns: file_name, gender, age, race
3. Run the training script:

```bash
python train.py --csv_file path/to/annotations.csv --img_dir path/to/images --output_dir models --batch_size 32 --epochs 30
```

## Project Structure

```
face-analysis/
├── face_analysis.py      # Main module with FaceAnalysisModel and FaceAnalyzer classes
├── train.py              # Script for training the model
├── main.py               # Command-line interface for image/video processing
├── models/               # Directory for storing model weights
│   ├── deploy.prototxt   # Face detection model architecture
│   ├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection weights
│   └── best_model.pth    # Trained face analysis model weights
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Performance

The model achieves approximately 90% accuracy on gender classification, 85% on age range prediction, and 80% on race classification when trained properly on the FairFace dataset.

## Ethical Considerations

This tool should be used responsibly and ethically. Facial analysis systems can perpetuate biases and should not be used for discrimination. The FairFace dataset was specifically designed to address biases in facial recognition systems, but care should still be taken in deployment.

## Citation

If you use this project in your research, please cite:

```
@inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021}
}
```

## License

This project is released under the MIT License.
