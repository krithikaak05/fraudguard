import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('/Users/krithikaannaswamykannan/fraud-pipeline/data/ieee-fraud-detection/train_transaction.csv')

features = [
    'TransactionAmt', 'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'dist1',
    'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
    'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

X = df[features].copy()
y = df['isFraud']

cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
for col in cat_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

X = X.fillna(-999)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}

print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
    use_label_encoder=False, eval_metric='auc', random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
xgb.save_model('/Users/krithikaannaswamykannan/fraud-pipeline/notebooks/model_xgboost.json')
results['XGBoost'] = round(xgb_auc, 4)
print(f"XGBoost AUC: {xgb_auc:.4f}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
results['Random Forest'] = round(rf_auc, 4)
print(f"Random Forest AUC: {rf_auc:.4f}")

print("\nTraining Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
results['Logistic Regression'] = round(lr_auc, 4)
print(f"Logistic Regression AUC: {lr_auc:.4f}")

best_model = max(results, key=results.get)
best_auc = results[best_model]
print(f"\nBest model: {best_model} with AUC {best_auc:.4f}")

with open('/Users/krithikaannaswamykannan/fraud-pipeline/notebooks/model_results.json', 'w') as f:
    json.dump({"results": results, "best_model": best_model, "best_auc": best_auc}, f)

print("All done. Results saved to model_results.json")
