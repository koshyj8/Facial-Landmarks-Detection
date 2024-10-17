# Facial Landmark Detection with Xception Architecture
Developed a facial landmark detection system leveraging XceptionNet and PyTorch. Implemented data preprocessing, GPU acceleration, and Matplotlib for visualization. The model accurately detects and overlays facial landmarks, suitable for applications in facial recognition and feature extraction.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Visualization](#results-visualization)
- [Contributing](#contributing)
- [License](#license)

## Features

- PyTorch implementation of facial landmark detection
- Detailed batch-level and epoch-level progress tracking
- PyTorch-based early stopping and learning rate scheduling
- Model checkpointing using `torch.save`
- Customizable data augmentation using PyTorch and Albumentation transforms

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- tqdm
- matplotlib
- albumentations
- opencv
- scikit-imahe

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/koshyj8/facial-landmarks-detection.git
   cd facial-landmark-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   ```
   pip install torch torchvision numpy matplotlib tqdm scikit-image albumentations opencv-python
   ```

## Usage

1. Prepare your dataset:
   Place your training and validation images in the dataset/ folder. The folder structure should be as follows:
2. Run the training script:
   jupyter notebook fld_xception.ipynb
3. Download the dataset from [https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset] and place it in the dataset/ folder
4. Train the model:
   Execute the training cells to train the model using the provided dataset. Modify hyperparameters like the number of epochs, batch size, and learning rate as needed.
5. Evaluate the model:
   After training, you can run the evaluation cells to test the model on validation images or your own test data.
6. Predict landmarks on a new image:
   To predict landmarks on a custom image, modify the prediction cells to specify the path to your input image and run the respective cells to visualize the output.
7. Visualize Results:
   After prediction, the landmarks will be drawn on the input image and displayed using Matplotlib. The output image will also be saved for further inspection.
   
## Project Structure
```
FLD/
├── dataset/
├── fld_xception.ipynb          # Main model training and evaluation file
├── model.pt                    # Pre-trained model weights
├── README.md                   # Project documentation
└── requirements.txt            # List of dependencies
```

## Model Architecture

- **Xception**: Pretrained Xception with a custom PyTorch regression head for landmark prediction

Both models output 136 values (68 landmarks * 2 coordinates) for each input image.

## Training Process

1. Data loading using PyTorch DataLoader
2. Training loop with PyTorch optimizers and batch-level progress tracking
3. Validation after each epoch using `torch.no_grad()`
4. Early stopping if no improvement in validation loss
5. Model checkpointing using `torch.save` to save the best model

## Evaluation Metrics

- Mean Squared Error (MSE) loss using PyTorch's `nn.MSELoss`

## Results Visualization

- Plot facial landmarks on images using `matplotlib`
- Compare original and predicted landmarks
- Visualize the training and validation loss curves using `matplotlib`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
