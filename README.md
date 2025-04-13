# face_analysis

# Face Analysis Project

This project implements a facial recognition system that detects faces and predicts gender, age range, and racial characteristics. It's based on the FairFace dataset approach described in the paper "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age" (Karkkainen & Joo, WACV 2021).

## Features

- Face detection using OpenCV's DNN face detector
- Multi-task learning for gender, age range, and race prediction
- Model trained on the FairFace dataset for balanced predictions across demographics
- Support for both image and video processing
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

3. Set up the project (downloads face detection models and optionally the FairFace dataset):
```bash
python project-setup.py  # Add --skip_dataset if you don't want to download the dataset yet
```

## Using the FairFace Dataset

This project is designed to work with the FairFace dataset, which provides a balanced distribution of faces across different demographics.

### Downloading the Dataset

To download the FairFace dataset, run:
```bash
python fairface_downloader.py --output_dir dataset
```

The script will:
1. Download the image dataset (~800MB)
2. Download the label files
3. Process and standardize the labels

### Training with FairFace

To train the model using the FairFace dataset:
```bash
python train.py --csv_file dataset/fairface_label_train.csv --val_csv_file dataset/fairface_label_val.csv --img_dir dataset --output_dir models --epochs 30
```

Training parameters:
- `--batch_size`: Number of images per batch (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 30)

## Usage

### Process an image

```bash
python main.py image --input path/to/image.jpg --output path/to/output.jpg --model models/best_model.pth --show
```

Arguments:
- `--input`: Path to the input image (required)
- `--output`: Path to save the annotated image (optional)
- `--model`: Path to the trained model file (default: models/best_model.pth)
- `--show`: Flag to display the result image

### Process a video

```bash
python main.py video --input path/to/video.mp4 --output path/to/output.mp4 --model models/best_model.pth
```

Arguments:
- `--input`: Path to the input video (required)
- `--output`: Path to save the processed video (optional)
- `--model`: Path to the trained model file (default: models/best_model.pth)


## Model Details

The face analysis model is a multi-task learning network that predicts:
- Gender: Male/Female
- Age Group: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
- Race/Ethnicity: White, Black, East Asian, Southeast Asian, Indian, Middle Eastern, Latino/Hispanic

The model architecture uses a ResNet-34 backbone followed by task-specific heads for each attribute.

## Performance

When properly trained on the FairFace dataset, the model achieves approximately:
- Gender classification: ~90% accuracy
- Age range prediction: ~85% accuracy
- Race classification: ~80% accuracy

Performance may vary based on training duration and dataset quality.

## Ethical Considerations

This tool should be used responsibly and ethically. Facial analysis systems can perpetuate biases and should not be used for discrimination. The FairFace dataset was specifically designed to address biases in facial recognition systems, but care should still be taken in deployment.

Some important ethical considerations:
1. Always obtain proper consent before analyzing someone's facial data
2. Be transparent about how the data is being used
3. Do not use the results for making important decisions that affect individuals
4. Be aware that demographic predictions are based on visual appearance and may not match self-identification

## Citation

If you use this project in your research, please cite the FairFace paper:

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
