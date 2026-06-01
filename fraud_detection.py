import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


def generate_dataset(n_samples=10000, random_state=42):
    np.random.seed(random_state)
    n_fraud = int(n_samples * 0.10)
    n_legit = n_samples - n_fraud

    def make(n, fraud):
        if fraud:
            return {
                'transaction_amount':    np.random.exponential(scale=2000, size=n),
                'time_of_day':           np.random.choice([0, 1, 2, 3, 22, 23], size=n),
                'ip_country_mismatch':   np.random.choice([0, 1], size=n, p=[0.2, 0.8]),
                'location_change':       np.random.choice([0, 1], size=n, p=[0.1, 0.9]),
                'failed_login_attempts': np.random.randint(3, 10, size=n),
                'account_age_days':      np.random.randint(1, 30, size=n),
                'num_transactions_24h':  np.random.randint(10, 50, size=n),
                'device_change':         np.random.choice([0, 1], size=n, p=[0.2, 0.8]),
                'avg_transaction_amt':   np.random.uniform(50, 300, size=n),
                'payment_method_code':   np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.2, 0.3, 0.4]),
                'is_fraud': np.ones(n, dtype=int)
            }
        else:
            return {
                'transaction_amount':    np.random.exponential(scale=300, size=n),
                'time_of_day':           np.random.randint(8, 22, size=n),
                'ip_country_mismatch':   np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
                'location_change':       np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
                'failed_login_attempts': np.random.randint(0, 2, size=n),
                'account_age_days':      np.random.randint(30, 3000, size=n),
                'num_transactions_24h':  np.random.randint(1, 8, size=n),
                'device_change':         np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
                'avg_transaction_amt':   np.random.uniform(100, 600, size=n),
                'payment_method_code':   np.random.choice([1, 2, 3, 4], size=n, p=[0.3, 0.3, 0.25, 0.15]),
                'is_fraud': np.zeros(n, dtype=int)
            }

    df = pd.concat(
        [pd.DataFrame(make(n_fraud, True)), pd.DataFrame(make(n_legit, False))],
        ignore_index=True
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    for col in ['transaction_amount', 'failed_login_attempts', 'account_age_days']:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df


def run_eda(df):
    print("\nDataset Shape  :", df.shape)
    print("Missing Values :\n", df.isnull().sum())
    print("\nClass Distribution:\n", df['is_fraud'].value_counts())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Online Fraud Detection — Exploratory Data Analysis", fontsize=15, fontweight='bold')

    counts = df['is_fraud'].value_counts()
    axes[0, 0].bar(['Legitimate', 'Fraud'], counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='white')
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    df[df['is_fraud'] == 0]['transaction_amount'].hist(ax=axes[0, 1], bins=50, alpha=0.7, color='#2ecc71', label='Legitimate')
    df[df['is_fraud'] == 1]['transaction_amount'].hist(ax=axes[0, 1], bins=50, alpha=0.7, color='#e74c3c', label='Fraud')
    axes[0, 1].set_title('Transaction Amount Distribution')
    axes[0, 1].set_xlabel('Amount')
    axes[0, 1].legend()

    mismatch = df.groupby(['ip_country_mismatch', 'is_fraud']).size().unstack()
    mismatch.plot(kind='bar', ax=axes[0, 2], color=['#2ecc71', '#e74c3c'], edgecolor='white')
    axes[0, 2].set_title('IP Country Mismatch vs Fraud')
    axes[0, 2].set_xticklabels(['No Mismatch', 'Mismatch'], rotation=0)
    axes[0, 2].legend(['Legitimate', 'Fraud'])

    axes[1, 0].boxplot(
        [df[df['is_fraud'] == 0]['failed_login_attempts'].dropna(),
         df[df['is_fraud'] == 1]['failed_login_attempts'].dropna()],
        labels=['Legitimate', 'Fraud'],
        patch_artist=True,
        boxprops=dict(facecolor='#3498db', color='white')
    )
    axes[1, 0].set_title('Failed Login Attempts')
    axes[1, 0].set_ylabel('Count')

    df[df['is_fraud'] == 0]['account_age_days'].hist(ax=axes[1, 1], bins=50, alpha=0.7, color='#2ecc71', label='Legitimate')
    df[df['is_fraud'] == 1]['account_age_days'].hist(ax=axes[1, 1], bins=50, alpha=0.7, color='#e74c3c', label='Fraud')
    axes[1, 1].set_title('Account Age Distribution')
    axes[1, 1].set_xlabel('Days')
    axes[1, 1].legend()

    sns.heatmap(df.corr(), ax=axes[1, 2], annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.5, annot_kws={'size': 7})
    axes[1, 2].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nEDA plots saved as eda_plots.png")


def preprocess(df):
    feature_cols = [c for c in df.columns if c != 'is_fraud']
    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    return X_imputed, y, feature_cols


def train_model(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nAccuracy  :", round(acc * 100, 2), "%")
    print("ROC-AUC   :", round(roc_auc, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fraud Detection Model Results', fontsize=14, fontweight='bold')

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues', linewidths=1,
                xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    axes[0].set_title(f'Confusion Matrix  (Accuracy {acc*100:.1f}%)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values(ascending=True).plot(kind='barh', ax=axes[2], color='#3498db', edgecolor='white')
    axes[2].set_title('Feature Importances')
    axes[2].set_xlabel('Score')

    plt.tight_layout()
    plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Model result plots saved as model_results.png")

    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(model, feature_names=feature_names, class_names=['Legitimate', 'Fraud'],
              filled=True, rounded=True, max_depth=2, fontsize=10, ax=ax,
              impurity=False, proportion=True)
    ax.set_title('Decision Tree Visualisation (max_depth=2)', fontsize=14, fontweight='bold')
    plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Decision tree plot saved as decision_tree.png")

    return acc, roc_auc


def predict_transaction(model, transaction, feature_names):
    row = pd.DataFrame([transaction])[feature_names]
    proba = model.predict_proba(row)[0][1]
    label = 'FRAUD' if proba >= 0.5 else 'LEGITIMATE'
    return {'label': label, 'fraud_probability': round(proba, 4)}


if __name__ == '__main__':

    print("Generating dataset...")
    df = generate_dataset(n_samples=10000)
    df.to_csv('transactions.csv', index=False)
    print(f"Shape: {df.shape}   Fraud rate: {df['is_fraud'].mean()*100:.1f}%")

    print("\nRunning EDA...")
    run_eda(df)

    print("\nPreprocessing...")
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating...")
    acc, auc = evaluate_model(model, X_test, y_test, feature_names)

    print("\nReal-time prediction demo:")

    sample_transactions = [
        {
            'transaction_amount': 8500, 'time_of_day': 2,
            'ip_country_mismatch': 1, 'location_change': 1,
            'failed_login_attempts': 7, 'account_age_days': 5,
            'num_transactions_24h': 30, 'device_change': 1,
            'avg_transaction_amt': 150, 'payment_method_code': 4
        },
        {
            'transaction_amount': 250, 'time_of_day': 14,
            'ip_country_mismatch': 0, 'location_change': 0,
            'failed_login_attempts': 0, 'account_age_days': 730,
            'num_transactions_24h': 2, 'device_change': 0,
            'avg_transaction_amt': 300, 'payment_method_code': 2
        }
    ]

    for i, txn in enumerate(sample_transactions, 1):
        result = predict_transaction(model, txn, feature_names)
        print(f"\nTransaction #{i}")
        print(f"  Amount        : {txn['transaction_amount']}")
        print(f"  IP Mismatch   : {'Yes' if txn['ip_country_mismatch'] else 'No'}")
        print(f"  Failed Logins : {txn['failed_login_attempts']}")
        print(f"  Account Age   : {txn['account_age_days']} days")
        print(f"  Prediction    : {result['label']}  (fraud prob: {result['fraud_probability']*100:.1f}%)")

    print(f"\nDone.  Accuracy={acc*100:.1f}%   ROC-AUC={auc:.3f}")
