import numpy as np
import pickle
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

file_path1 = "/mnt/c/Users/monta/Desktop/ReprésentationsCleanware/pe_feats_mal.csv"
file_path2 = "/mnt/c/Users/monta/Desktop/ReprésentationsCleanware/pe_feats_clean.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# On enlève la première colonne, celle des familles de Malwares
df1 = df1.drop(df1.columns[0], axis=1)

# On enlève les noms/les hashs
df1 = df1.drop(df1.columns[0], axis=1)
df2 = df2.drop(df2.columns[0], axis=1)

# On ajoute le nom des features
first_row = [f'F{i}' for i in range(1, 120)]
df1.columns = first_row
df2.columns = first_row

# On ajoute une colonne avec les 'Labels', 0 ou 1 selon si c'est un cleanware ou un malware
df1.insert(0, 'Label', 1)
df2.insert(0, 'Label', 0)

# Concaténer les deux DataFrames l'un après l'autre
df = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df)

# Imprimer les labels avant le shuffle
labels_before_shuffle = df['Label'].unique()
print("Labels before shuffle:", labels_before_shuffle) 

# On mélange (shuffle)
df = df.sample(frac=1).reset_index(drop=True)

# On remplace les NaN par un zéro
df = df.fillna(0)

# Calculer le nombre de lignes pour 80%
n = int(len(df) * 0.8)

# Sélectionner les premiers 80%
df_train = df.iloc[:n]

# Calculer le nombre de lignes pour 20%
m = int(len(df) * 0.2)

# Sélectionner les derniers 20%
df_test = df.iloc[-m:]

X_train = df_train.drop('Label', axis=1)
y_train = df_train['Label']

X_test = df_test.drop('Label', axis=1)
y_test = df_test['Label']

# Assurez-vous que y_train et y_test sont des vecteurs 1D
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(X_test)
print(y_test)

# Recherche des indices où y_train est None
indices_to_remove = np.where(np.isnan(y_train))

# Suppression des lignes correspondantes dans X_train et y_train
X_train = np.delete(X_train, indices_to_remove, axis=0)
y_train = np.delete(y_train, indices_to_remove)

# Recherche des indices où y_test est None
indices_to_remove = np.where(np.isnan(y_test))

# Suppression des lignes correspondantes dans X_test et y_test
X_test = np.delete(X_test, indices_to_remove, axis=0)
y_test = np.delete(y_test, indices_to_remove)

# Création et entraînement du modèle
GB_model = GradientBoostingClassifier(n_estimators=20, max_depth=5, learning_rate=0.2, min_samples_leaf=20, random_state=42)

GB_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = GB_model.predict(X_test)
y_pred1 = GB_model.predict(X_train)

# Calcul de l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
accuracy_2 = accuracy_score(y_train, y_pred1)
print("Accuracy de train:", accuracy_2)

# Calcul des métriques
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, GB_model.predict_proba(X_test)[:, 1])
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Affichage des résultats
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Enregistrer le modèle avec pickle
model_filename = 'GB_model_pe_feats_détection.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(GB_model, file)
