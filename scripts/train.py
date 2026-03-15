import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

print("Starte KI-Training...")

df_fights = pd.read_csv("data/Fights.csv")
df_stats = pd.read_csv("data/Fighters Stats.csv")


df_fights = df_fights[df_fights['Result_1'].isin(['W', 'L'])]
df_fights['Target'] = df_fights['Result_1'].apply(lambda x: 1 if x == 'W' else 0)

# Daten zusammenführen
stats_f1 = df_stats.copy()
stats_f1.columns = [f"{col}_F1" for col in stats_f1.columns]
df_model = df_fights.merge(stats_f1, left_on='Fighter_Id_1', right_on='Fighter_Id_F1', how='left')

stats_f2 = df_stats.copy()
stats_f2.columns = [f"{col}_F2" for col in stats_f2.columns]
df_model = df_model.merge(stats_f2, left_on='Fighter_Id_2', right_on='Fighter_Id_F2', how='left')

numeric_cols = df_model.select_dtypes(include=['number']).columns.tolist()

features = [col for col in numeric_cols if col not in ['Target', 'KD_1', 'KD_2', 'STR_1', 'STR_2']]

df_model = df_model.dropna(subset=features)

X = df_model[features]
y = df_model['Target']

# Train / Test Split (80% zum Lernen, 20% zum Prüfen)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Trainiere XGBoost mit {len(X_train)} Kämpfen und {len(features)} Features...")
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Kurzer Test
accuracy = model.score(X_test, y_test)
print(f"Training fertig. Genauigkeit auf Testdaten: {accuracy * 100:.2f}%")

# Modell speichern
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.joblib")
print("Modell erfolgreich in 'models/xgb_model.joblib' gespeichert!")