
# DarkNet_Tor_Classifier.py
# Σενάριο: Εντοπισμός Tor vs Non-Tor traffic

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# ===============================
# 1. Φόρτωση Dataset
# ===============================
raw_url = "https://raw.githubusercontent.com/kdemertzis/EKPA/main/Data/DarkNet.csv"
df = pd.read_csv(raw_url)

print("Διαστάσεις dataset:", df.shape)
print("Στήλες:", df.columns.tolist()[:10], "...")

# ===============================
# 2. Δημιουργία Label
# ===============================
label_col = df.columns[-1]
df['label'] = df[label_col].astype(str).str.lower().apply(
    lambda x: 'Tor' if 'tor' in x else 'Non-Tor'
)

# ===============================
# 3. Επιλογή Χαρακτηριστικών
# ===============================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols].fillna(df[num_cols].median())
y = df['label'].map({'Tor': 1, 'Non-Tor': 0})

# ===============================
# 4. Split Train/Test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 5. Balancing με SMOTE
# ===============================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ===============================
# 6. Scaling
# ===============================
scaler = StandardScaler()
X_train_res_sc = scaler.fit_transform(X_train_res)
X_test_sc = scaler.transform(X_test)

# ===============================
# 7. Εκπαίδευση Μοντέλων
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_train_res_sc, y_train_res),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ).fit(X_train_res, y_train_res),
    "XGBoost": xgb.XGBClassifier(
        use_label_encoder=False, eval_metric='logloss',
        max_depth=6, n_estimators=100, n_jobs=-1, random_state=42
    ).fit(X_train_res, y_train_res)
}

# ===============================
# 8. Αξιολόγηση
# ===============================
results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        y_pred = model.predict(X_test_sc)
    else:
        y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred) το 
    results[name] = f1
    print("\n---", name, "---")
    print("F1-score:", round(f1, 4))
    print(classification_report(y_test, y_pred, target_names=["Non-Tor","Tor"]))

# ===============================
# 9. Αποθήκευση Καλύτερου Μοντέλου
# ===============================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print(f"\n✅ Καλύτερο Μοντέλο: {best_model_name}")
print("Τα αρχεία best_model.joblib και scaler.joblib αποθηκεύτηκαν επιτυχώς.")

