# 🛍️ Amazon Review Sentiment Analysis using BERT

## 📌 Project Overview

This project implements a **BERT-based Sentiment Analysis system** for Amazon product reviews using the HuggingFace Transformers library and PyTorch.

The model classifies reviews into:

- **Positive (1)** → Rating ≥ 4  
- **Negative (0)** → Rating < 4  

The model is trained using a custom training loop with:
- Gradient accumulation
- Learning rate scheduling
- Mixed precision training (AMP)
- Stratified dataset splitting
- Automatic Drive model saving (Colab integration)

---

## 🧠 Model Used

- **Model:** `bert-base-uncased`
- **Framework:** PyTorch
- **Library:** HuggingFace Transformers
- **Task:** Binary Text Classification

---

## 📂 Dataset

- Input File: `amazon_review_revised.csv`
- Automatically detects:
  - Text column (`review_text`, `review`, `text`, etc.)
  - Rating column (`rating`, `stars`, `overall`, etc.)

### Label Mapping

```python
if rating >= 4:
    label = 1   # Positive
else:
    label = 0   # Negative
```

---

## ⚙️ Key Features

- ✅ Automatic column detection  
- ✅ Binary or Multi-class configuration  
- ✅ Train / Validation / Test split (Stratified)  
- ✅ Custom PyTorch Dataset & DataLoader  
- ✅ Gradient Accumulation  
- ✅ Learning Rate Scheduler with Warmup  
- ✅ Mixed Precision Training (GPU optimized)  
- ✅ Model checkpoint saving to Google Drive  
- ✅ Automatic model loading if already trained  

---

## 🏗️ Project Workflow

1. Mount Google Drive  
2. Load & clean dataset  
3. Map ratings to sentiment labels  
4. Train/Validation/Test split  
5. Tokenization using BERT tokenizer  
6. Model training  
7. Validation after each epoch  
8. Save best model based on F1-score  
9. Load model for inference  

---

## 🔧 Hyperparameters

| Parameter | Value |
|------------|--------|
| Model | bert-base-uncased |
| Max Length | 128 |
| Batch Size | 16 |
| Epochs | 4 |
| Learning Rate | 3e-5 |
| Gradient Accumulation | 2 |
| Optimizer | AdamW |
| Scheduler | Linear Warmup |

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score (Weighted)  
- Confusion Matrix  
- Classification Report  

Validation is performed after each epoch and the best model (based on F1-score) is saved.

---

## 💾 Model Saving

The trained model and tokenizer are saved to:

```
/content/drive/MyDrive/amazon_sentiment/saved_model_bert
```

If the model already exists, it loads automatically unless `FORCE_TRAIN = True`.

---

## 🖥️ Hardware Support

- Automatically detects CUDA (GPU)
- Uses:
  - Mixed Precision (AMP)
  - Pin memory optimization
  - Multi-worker DataLoader

---

## 📦 Libraries Used

- transformers  
- torch  
- pandas  
- numpy  
- scikit-learn  
- tqdm  
- matplotlib  

---

## 🚀 Future Improvements

- Add Gradio web interface for live predictions  
- Deploy using Streamlit / FastAPI  
- Add hyperparameter tuning  
- Implement early stopping  
- Support multi-class sentiment classification  

---

## 🎯 Objective

To build a production-ready BERT sentiment classification pipeline with efficient training, evaluation, and model persistence for large-scale Amazon review datasets.
