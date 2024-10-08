import numpy as np
import pickle
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


file_path = "/mnt/c/Users/monta/Desktop/BODMAS2/Features_Ember.csv"

df = pd.read_csv(file_path)


#On enlève les noms/les hashs
df = df.drop(df.columns[0], axis=1)

# Réinitialiser l'index pour récupérer la colonne
df = df.reset_index()


#On ajoute le nom des features
first_row = ['Label'] + [f'F{i}' for i in range(1, 2352)]
df.columns = first_row

# Imprimer les labels avant le shuffle
labels_before_shuffle = df['Label'].unique()
print("Labels before shuffle:", labels_before_shuffle) 


#On mélange (shuffle)
df = df.sample(frac=1).reset_index(drop=True)


#On remplace les NaN par un zéro
df = df.fillna(0)

# Calculer le nombre de lignes pour 70%
n = int(len(df) * 0.8)

# Sélectionner les premiers 70%
df_train = df.iloc[:n]

# Calculer le nombre de lignes pour 30%
m = int(len(df) * 0.2)

# Sélectionner les derniers 30%
df_test = df.iloc[-m:]


X_train = df_train.drop('Label', axis=1)
y_train = df_train['Label']

X_test = df_test.drop('Label', axis=1)
y_test = df_test['Label']


print(X_test)
print(y_test)

# Recherche des indices où y_train est None
indices_to_remove = y_train.index[y_train.isnull()]

# Suppression des lignes correspondantes dans X_train et y_train
X_train = X_train.drop(indices_to_remove)
y_train = y_train.drop(indices_to_remove)

# Recherche des indices où y_test est None
indices_to_remove = y_test.index[y_test.isnull()]

# Suppression des lignes correspondantes dans X_test et y_test
X_test = X_test.drop(indices_to_remove)
y_test = y_test.drop(indices_to_remove)

#Création et entraînement du modèle


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
roc_auc = roc_auc_score(y_test, GB_model.predict_proba(X_test), multi_class='ovo')
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
model_filename = 'GB_model_ember.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(GB_model, file)