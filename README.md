# Multi-Label Emotion Classification with Transformers

This project implements a **multi-label emotion classifier** using transformer models like **distilbert-base-uncased** and **DistilBERT**. It detects five emotions — `anger`, `fear`, `joy`, `sadness`, and `surprise` — from English text.

---

## Project Highlights

- Fine-tuned **`distilbert-base-uncased`** on a cleaned emotion dataset.
- Supports **multi-label classification** using `BCEWithLogitsLoss` and **Focal Loss**.
- Uses **stratified splitting** via `iterative-stratification` for balanced label distribution.
- Includes **data augmentation** using `nlpaug` for underrepresented classes.
- **Threshold optimization** to improve per-class F1 scores.
- Early stopping and gradient accumulation for stable training.

---

## 📁 Project Structure

Multi-Label-Emotion-Detection/
├── main.py # Optional script entry point
├── train.py # Full training pipeline
├── emotion_classifier/
│ └── src/
│ ├── prepare_data.py # Data cleaning
│ ├── data_loader.py # Dataset + loader utilities
│ ├── model_utils.py # Model architecture & loader
│ ├── predictor.py # (Optional) Inference helper
├── data/
│ ├── raw/ # Raw CSV data
│ ├── processed/ # Cleaned datasets
│ └── valid/ # Validation/test samples
├── notebooks/ # Jupyter exploration
├── archiv/ # Pickled encodings + trained models
├── requirements.txt
└── README.md

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Multi-Label-Emotion-Detection.git
cd Multi-Label-Emotion-Detection
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
   **_If using optional features_**
   pip install nlpaug iterative-stratification
```

4. **Training**

```bash
python train.py
```

This will:

Clean and tokenize data
Optionally augment text
Train the model with early stopping
Optimize prediction thresholds
Save best models to best_model.pt and trained_model.pt

5. **Results**

| Metric          | Value                               |
| --------------- | ----------------------------------- |
| Best Val F1     | \~0.85                              |
| Test F1 (macro) | \~0.85                              |
| Model           | `distilbert-base-uncased`           |
| Emotions        | anger, fear, joy, sadness, surprise |

---

## 🔧 Inference via `main.py`

The project includes a ready-to-use CLI script `main.py` for predicting emotions from single texts or entire CSV files.

### Option 1: Predict a single sentence

```bash
python main.py --text "I'm feeling anxious and overwhelmed but also a bit hopeful."
```

```

Example output:
🔍 Input: I'm feeling anxious and overwhelmed but also a bit hopeful.
✅ Emotions: ['fear', 'joy']

```

### Option 2: Predict from a CSV file

The CSV must contain a column named text.

```bash
python main.py --csv data/valid/test_from_track.csv
```

Example output:
✅ Predictions saved to data/valid/test_from_track_predictions.csv

| text                                     | predicted_emotions    |
| ---------------------------------------- | --------------------- |
| I screamed so loud my voice cracked      | ['anger', 'surprise'] |
| It reminded me of my grandmother's house | ['joy']               |

**_Acknowledgements_**

- HuggingFace transformers
- iterative-stratification by scikit-multilearn
- nlpaug for data augmentation
- TQDM and PyTorch
