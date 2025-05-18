# Face Detection using SVM

This project implements a face detection pipeline using Support Vector Machine (SVM) as the classifier. It is inspired by a 2008 research paper but improves upon it by using modern datasets and tools for faster and more accurate results.

---

## 📖 Project Background

### Original Paper Approach:
- Used **1066 grayscale human face images** focused mainly on the eyes.
- Generated **2132 positive samples** (20x20 pixels, eyes marked).
- Negative samples were 3 to 10 times the number of positive samples.
- Used an **RBF kernel SVM** with sigma between 800-1600 and C=10.

### My Approach:
- Used the **`fetch_lfw_people` dataset** from Scikit-Learn with **13233 full face images** as positive samples.
- Generated **30000 negative samples** using Scikit-Image's `PatchExtractor` on images like `camera`, `moon`, `text`, etc.
- Combined positive and negative samples to create a dataset of **43233 samples**.
- Extracted features using the **HOG (Histogram of Oriented Gradients)** method.
- Trained both **Linear SVM** and **Gaussian (RBF) SVM** models:
    - Linear SVM used `LinearSVC` with best `C=1` after Grid Search.
    - Gaussian SVM followed paper settings (RBF kernel, bootstrapping).

---

## 📂 Project Structure
```
├── face_detection_SVM.ipynb  # Main Jupyter Notebook with full implementation
├── /data/                   # Directory containing positive and negative image samples
└── /models/                 # (Optional) Directory to save trained models
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/face_detection_SVM.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the notebook:
```bash
jupyter notebook face_detection_SVM.ipynb
```

---

## 📚 Dependencies
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Scikit-Learn
- Scikit-Image
- Matplotlib

Install them with:
```bash
pip install opencv-python numpy scikit-learn scikit-image matplotlib
```

---

## 📈 Results & Performance

### Linear SVM:
- Training Time: **~1-2 minutes** (Google Colab)
- Detected 41 patches correctly, **0 false positives** on masked faces.
- Confusion Matrix:
```
TP: 5915 | FP: 27  
FN: 50   | TN: 2655
```

### Gaussian (RBF) SVM with Bootstrapping (5 Iterations):
- Training Time: **~45 minutes** (Google Colab)
- Support Vectors per Iteration: [2069, 2044, 2008, 2044, 1989]
- Accuracy: **99.63% ± 0.02%**
- Confusion Matrix:
```
TP: 5941 | FP: 1  
FN: 32   | TN: 2673
```
- Only **1 false positive** detected. Incredible performance!

---

## 📖 HOG Feature Extraction
The Histogram of Oriented Gradients (HOG) was used for feature extraction, significantly improving detection performance. Key steps include:
1. Pre-normalizing images to reduce illumination effects.
2. Computing gradient orientations.
3. Building orientation histograms per cell.
4. Normalizing and flattening into feature vectors.

---

## 📄 License
This project is licensed under the MIT License.

---

*Feel free to fork the repository and contribute! 😊*
