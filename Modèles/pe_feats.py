import numpy as np
import pickle
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

file_path = "/mnt/c/Users/monta/Desktop/BODMAS2/pe_feats.csv"

df = pd.read_csv(file_path)

#On enlève les noms/les hashs
df = df.drop(df.columns[1], axis=1)

#On ajoute le nom des features
first_row = ['Label'] + [f'F{i}' for i in range(1, 120)]
df.columns = first_row

#On mélange (shuffle)
df = df.sample(frac=1).reset_index(drop=True)

#On remplace les NaN par un zéro
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

# Vérification des classes uniques dans y_train et y_test
unique_classes_train = np.unique(y_train)
unique_classes_test = np.unique(y_test)
print("Unique classes in y_train:", unique_classes_train)
print("Unique classes in y_test:", unique_classes_test)

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

# Ajustement de roc_auc_score
if len(unique_classes_test) == len(unique_classes_train):
    roc_auc = roc_auc_score(y_test, GB_model.predict_proba(X_test), multi_class='ovr')
else:
    print("Warning: The number of classes in the test set does not match the number of classes in the training set.")
    roc_auc = None

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Affichage des résultats
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
if roc_auc is not None:
    print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Enregistrer le modèle avec pickle
model_filename = 'GB_model_pe_feats.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(GB_model, file)
