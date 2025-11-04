# Breast Cancer Classification using a Simple Neural Network

This project implements a **neural network from scratch** (no TensorFlow or PyTorch) to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the **Breast Cancer Wisconsin Dataset**.  

Only two features were used as inputs:
- `concave points_worst`  
- `perimeter_worst`  

---

## Overview
- Implemented a **fully connected neural network** using a custom-built autograd engine (`Value` class).  
- Performed **manual forward and backward propagation**.  
- Trained the model using **Mean Squared Error (MSE)** loss.  
- Evaluated performance using **scikit-learn** metrics.

---

## Training Configuration
| Parameter | Value |
|------------|--------|
| Epochs | 1000 |
| Learning Rate | Dynamic (decays with epochs) |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Manual Gradient Descent |

---

## Evaluation Results
| Metric | Score |
|---------|--------|
| **Train Accuracy** | 94.51% |
| **Test Accuracy** | 93.86% |
| **Precision** | 93.48% |
| **Recall** | 91.49% |
| **F1 Score** | 92.47% |
| **ROC AUC** | 0.987 |

### Confusion Matrix
- **True Positives (TP):** 43  
- **True Negatives (TN):** 64  
- **False Positives (FP):** 3  
- **False Negatives (FN):** 4  

These results indicate the model can **accurately distinguish between benign and malignant tumors**, even with only two input features.

---

##  Visualizations
- **Training Loss Curve** — shows steady loss reduction over epochs.  
- **Confusion Matrix** — displays classification performance visually.  
- **ROC Curve** — demonstrates strong separability (AUC ≈ 0.987).  

---

## Libraries Used
- `pandas`
- `numpy`  
- `matplotlib`  
- `scikit-learn`  

---

## How to Run

1. Install the required dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn
   
2. Open and run the Jupyter Notebook

3. Execute all cells to train the model and view the evaluation metrics.


## Acknowledgment

This project is inspired by **Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)** — a minimalistic autograd and neural network library written from scratch.  
His tutorial served as the foundation for understanding backpropagation, gradient descent, and building neural networks from first principles.
Portions of the autograd engine are adapted from Karpathy’s micrograd project.
