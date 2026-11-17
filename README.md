# Malaria Cell Detection (Classical Machine Learning)

A machine learning project that classifies microscopic blood cell images as **Parasitized** (malaria-infected) or **Uninfected**, using handcrafted image features and classical ML models.

## Dataset
- Cell images in two classes: Uninfected and Parasitized (Kaggle "Cell Images for Detecting Malaria").
- 7,000 images used (3,500 per class), resized to 64x64.
- 80/20 train/test split (`random_state=42`).

## Feature Extraction
Each image is converted into a 519-dimensional feature vector:
- 3D HSV color histogram (8x8x8 bins)
- Mean and standard deviation of the RGB channels
- Edge sharpness (variance of the Laplacian)

## Pipeline
1. Load images and extract features (OpenCV)
2. Check for missing values and duplicates
3. Train/test split
4. Train models before and after `StandardScaler`
5. Hyperparameter tuning with `GridSearchCV` (3-fold CV)
6. Evaluate with classification reports and confusion matrices
7. Save trained models with `joblib`

## Results (test set, 1,400 images)

| Model | Accuracy |
|---|---|
| Logistic Regression (scaled) | 93% |
| SVM (GridSearchCV, scaled) | 92% |
| Decision Tree (GridSearchCV, max_depth=5) | **95%** |

Scaling improved Logistic Regression from 82% to 93% and SVM from 82% to 92%. Decision Tree is unaffected by scaling.

## How to Run
```bash
pip install -r requirements.txt
```
1. Download the dataset and place it at `archive/cell_images/`
2. Open `Finalpro.ipynb` and run all cells

## Tech Stack
Python, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, joblib

## Author
Islam Mehanna