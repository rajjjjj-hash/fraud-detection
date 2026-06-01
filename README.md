# Online Fraud Detection using Machine Learning

A machine learning project that detects fraudulent online transactions in real time using a Decision Tree classifier.

---

## Project Summary

With the rise of digital payments, online fraud has become a serious threat. This project builds an end-to-end ML pipeline that analyses transaction behaviour and flags suspicious activity before any financial damage occurs.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| scikit-learn | ML model and metrics |
| pandas / numpy | Data handling |
| matplotlib / seaborn | Visualisations |

---

## Project Structure

```
fraud_detection/
├── fraud_detection.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
python fraud_detection.py
```

---

## Pipeline

```
Raw Data  →  EDA  →  Preprocessing  →  Train  →  Evaluate  →  Predict
```

1. Data Generation — 10,000 synthetic transactions with realistic fraud patterns
2. EDA — Distribution plots, correlation heatmap, class analysis
3. Preprocessing — Missing value imputation, 80:20 stratified split
4. Model Training — Decision Tree with depth regularisation and class balancing
5. Evaluation — Accuracy, F1, ROC-AUC, confusion matrix, feature importance
6. Real-time Prediction — Classifies any new transaction instantly

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~94% |
| ROC-AUC | 0.97+ |
| Precision | ~91% |
| Recall | ~89% |
| F1-Score | ~90% |

---

## Key Fraud Signals

- Unusually high transaction amounts
- IP country mismatch
- Multiple failed login attempts
- New accounts making large transfers
- Sudden device or location changes

---

## Sample Output

```
Transaction #1:
  Amount        : 8500
  IP Mismatch   : Yes
  Failed Logins : 7
  Account Age   : 5 days
  Prediction    : FRAUD  (fraud prob: 92.3%)

Transaction #2:
  Amount        : 250
  IP Mismatch   : No
  Failed Logins : 0
  Account Age   : 730 days
  Prediction    : LEGITIMATE  (fraud prob: 3.1%)
```

---

## What I Learned

- Handling class imbalance using class_weight='balanced'
- Preventing overfitting with max_depth regularisation
- Feature engineering for fraud detection use cases
- Building real-time prediction functions
- Interpreting models through feature importance and decision tree plots

---

## Future Improvements

- Compare with Random Forest and XGBoost
- Add SMOTE for oversampling the minority class
- Deploy as a Flask or FastAPI web service
- Connect to a live transaction stream

---

## Author

**Raj Rajeswar Singh Baghel**
[LinkedIn](https://www.linkedin.com/in/rajrajeswar-singh-baghel) | [GitHub](https://github.com/rajjjjj-hash)

---

> Made by [Raj Rajeswar Singh Baghel](https://www.linkedin.com/in/rajrajeswar-singh-baghel)
