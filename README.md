# Pet Classifier - Transfer Learning Project

This project implements a transfer learning approach to classify images of cats and dogs using TensorFlow and a pre-trained model.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create the following directory structure:
```
data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

5. Download the dataset from [Microsoft's Cat and Dog dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) and extract the images into the appropriate directories.

## Project Structure

- `train.py`: Main script for training the model
- `data/`: Directory containing training and test datasets
- `models/`: Directory for saving trained models
- `requirements.txt`: Project dependencies

## Usage

1. Prepare your dataset in the data/ directory
2. Run the training script:
```bash
python train.py
```

## Dataset

The project uses the Cats vs Dogs dataset from Microsoft. The dataset contains thousands of images of cats and dogs for training a binary classifier.
