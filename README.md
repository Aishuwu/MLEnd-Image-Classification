# Image Classification: Rice vs Chips using Classical Machine Learning

## Introduction

This project addresses a binary image classification task aimed at distinguishing between two food categories: rice and chips. The classification is performed using traditional machine learning models, supported by engineered features derived from color and texture information. The pipeline includes image preprocessing, feature extraction, model training, evaluation, and deployment.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Improvements](#improvements)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- Classification of food images as either rice or chips.
- Feature extraction from images using:
  - **Color Component**: Yellow channel from HSV space.
  - **Texture Features**: GLCM dissimilarity and correlation.
- Two classification models:
  - Linear Support Vector Classifier (LinearSVC)
  - Random Forest Classifier
- Image standardization and resizing.
- Evaluation using accuracy metrics and confusion matrix.
- Predictive deployment function for inference on new images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aishuwu/rice-vs-chips-classification.git
   cd rice-vs-chips-classification
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Mount Google Drive and download the dataset using:
   ```python
   from mlend import download_yummy_small
   baseDir = download_yummy_small(save_to='/content/drive/MyDrive/Data/MLEnd')
   ```

## Usage

1. Run the Jupyter notebook or scripts to:
   - Load the dataset using `yummy_small_load`.
   - Preprocess images (square cropping, resizing to 200x200).
   - Extract features (yellow component and GLCM metrics).
   - Standardize features.
   - Train and evaluate models.
   - Visualize confusion matrix and accuracy.

2. Use the deployed function:
   ```python
   predicting_rice_or_chips('path/to/image.jpg')
   ```

## Dataset

- **Source**: Subset from MLEnd's `yummy_small` dataset.
- **Content**: Color images of dishes labeled as either containing `rice` or `chips`.
- **Size**:
  - 70 training images
  - 29 testing images
- **Labels**:
  - `0` for chips
  - `1` for rice

Preprocessing includes:
- Square cropping and resizing to 200x200 pixels.
- Yellow color channel isolation from HSV.
- GLCM feature extraction for texture.

## Methodology

- **Data Splitting**: Stratified split using `Benchmark_A` ensures balanced training and testing sets.
- **Feature Engineering**:
  - Yellow component (HSV color space)
  - GLCM dissimilarity and correlation
- **Standardization**: Based on training set mean and std deviation.
- **Modeling**: Training two models:
  - LinearSVC
  - Random Forest (n_estimators=5, max_depth=3)
- **Evaluation**:
  - Accuracy metrics
  - Confusion matrix

## Modeling

- **Linear Support Vector Classifier (LinearSVC)**:
  - Efficient for linearly separable binary classification.
  - Training Accuracy: ~62.9%
  - Testing Accuracy: ~58.6%

- **Random Forest Classifier**:
  - Ensemble method robust to overfitting.
  - Deployed for final inference.

## Evaluation

- Confusion matrix reveals class-wise performance:
  - Better prediction accuracy for chips vs rice.
- Accuracy metrics calculated for:
  - Training set
  - Testing set
  - Class-wise accuracy and misclassification rates

## Improvements

Future enhancements may include:
- Hyperparameter tuning
- Additional color/texture features
- Model ensemble techniques
- Neural network-based classifiers for deeper learning

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
scikit-image
mlend
librosa
tqdm
```

Install them with:
```bash
pip install -r requirements.txt
```

## License

This project is provided for educational and research purposes only.
