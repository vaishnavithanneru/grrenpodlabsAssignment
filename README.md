# Avocado Disease Classification Using Machine Learning and Deep Learning

## Project Overview

This project focuses on classifying orange leaf images into four categories: **Black Spot, Canker, Fresh, and Greening** using classical machine learning and deep learning techniques.

The project is implemented in Python using Visual Studio Code (VS Code). It compares a classical machine learning approach using image features and Support Vector Machine (SVM) with a deep learning approach using MobileNetV2.

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scikit-image
- PyTorch
- Pillow
- Joblib
- Visual Studio Code

## Project Structure

```text
greenpodlabsassignment/
├── avocado_assignment.py
├── requirements.txt
├── Orange Dataset/
│   ├── train/
│   │   ├── blackspot/
│   │   ├── canker/
│   │   ├── fresh/
│   │   └── greening/
│   └── test/
│       ├── blackspot/
│       ├── canker/
│       ├── fresh/
│       └── greening/
├── models/
└── results/
```

## Approaches Implemented

### 1. Classical Machine Learning

- Extracts color histogram features from images using the CIELAB color space.
- Extracts texture features using Local Binary Patterns (LBP).
- Uses feature scaling and a Support Vector Machine (SVM) classifier.
- Saves the trained model for future predictions.

### 2. Deep Learning Using MobileNetV2

- Uses the pretrained MobileNetV2 architecture.
- Applies image preprocessing and data augmentation.
- Trains the classification layer to recognize the four orange image categories.
- Uses validation loss and early stopping to help prevent overfitting.
- Saves the trained model for future predictions.

## Exploratory Data Analysis

The project includes exploratory data analysis to examine the dataset, visualize class distributions, and display sample images.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The results are saved in the `results/` directory for comparison.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Visual Studio Code
- The orange image dataset arranged in the structure shown above

### Step 1: Open the Project

Open the `greenpodlabsassignment` folder in Visual Studio Code.

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python3 avocado_assignment.py
```

Run this command from the project root directory.

## Prediction

The script supports image prediction through its command-line interface. To predict an image, use:

```bash
python3 avocado_assignment.py --predict "path/to/orange_image.jpg"
```

Replace the image path with the actual path to the image you want to classify.

## Train the Models

Run the following command from the project root directory to train the models:
```bash
python3 train_models.py
```

This command trains the models and saves the trained model files required for image prediction. Ensure that the dataset is organized correctly before starting the training process.

## Install Streamlit and Dependencies

Install the required Python packages using:

```bash
python3 -m pip install -r requirements.txt
```

If Streamlit is not included in requirements.txt, install it separately:
```bash
python3 -m pip install streamlit
```
## Run the Streamlit Application

After training the models and installing the dependencies, start the Streamlit application:
```bash
python3 -m streamlit run app.py
```
Replace app.py with the actual filename of your Streamlit application if it is different.

## Test Image Predictions

Once the application starts, open the local URL displayed in the terminal, usually:

http://localhost:8501

Upload an image to view the model's predicted class and the results displayed by the application.



## Output

The project generates trained model files in the `models/` directory and evaluation outputs, such as plots and comparison results, in the `results/` directory.

## Development Environment

This project was developed and executed locally using **Visual Studio Code (VS Code)**. The Python script, dataset, dependencies, trained models, and generated results are organized within the project directory.

## Conclusion

This project compares classical machine learning and deep learning approaches for orange disease image classification. It demonstrates image feature extraction, model training, performance evaluation, and image prediction in a Python-based computer vision workflow.
### Streamlit Application

The Streamlit application allows users to upload an image and predict its class. It displays the uploaded image along with the model's prediction results, making it easy to test and understand the model's performance.

**Live Demo:** http://192.168.0.101:8501/

