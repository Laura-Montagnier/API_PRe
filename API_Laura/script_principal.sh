#!/bin/bash

# Définir le chemin vers le répertoire des exécutables
dir_path="/home/laura/API_PRe/API_Laura/Fichiers_exécutables/"

# Exécuter le script Python pour les Grayscales avec le chemin en argument
python3 représentation_grayscale.py "$dir_path"

# Charger le modèle à partir du fichier .pkl
python3 << END
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('/home/laura/API_PRe/Modèles')
from convnet import Convnet

Labels = ['benjamin','berbew','ceeinject','dinwod','ganelp','gepys','mira','sfone','sillyp2p','upatre','wabot','wacatac','musecador']

# Chemin complet vers le modèle .pkl
model_path = '/home/laura/API_PRe/Modèles/cnn_model_grayscale.pkl'

# Charger le modèle
with open(model_path, 'rb') as file:
	cnn_model_grayscale = pickle.load(file)

# Chemin complet vers le répertoire contenant les fichiers Grayscales
grayscales_dir = "/home/laura/API_PRe/API_Laura/Résultats/grayscale"

# Parcourir les fichiers dans le répertoire Grayscales
for filename in os.listdir(grayscales_dir):
	file_path = os.path.join(grayscales_dir, filename)
        
	# Charger les données à partir du fichier
	image = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
	data = np.array(image)
        
	# Reshape the data to match the input shape of the model
	data = data.reshape(1, 1, data.shape[0], data.shape[1])
	data = torch.tensor(data, dtype=torch.float32)

	# Faire des prédictions avec le modèle
	cnn_model_grayscale.eval()  # Mettre le modèle en mode évaluation

	with torch.no_grad():
		# Obtenir les sorties brutes du modèle
		outputs = cnn_model_grayscale(data)
		# Appliquer la fonction softmax pour obtenir les probabilités
		probabilities = F.softmax(outputs, dim=1)
		# Obtenir l'indice de la classe prédite
		idx = probabilities.argmax(axis=1).cpu().numpy()[0]
		# Afficher les probabilités et la prédiction
		print(f"La probabilité de prédiction est de : {probabilities.cpu().numpy()[0][0]}")
		if probabilities.cpu().numpy()[0][0]==1:
			print("Le Grayscale suffit.")
		print(f"Index de la classe prédite pour {filename}: {Labels[idx]}")


# Supposons que 'ma_variable' contienne la valeur que vous souhaitez récupérer
		ma_variable = str(probabilities.cpu().numpy()[0][0])

# Écriture de la valeur dans un fichier
		with open('mon_fichier.txt', 'w') as f:
			f.write(ma_variable)
END

valeur_recuperee=$(< mon_fichier.txt)

if [ "$valeur_recuperee" == "1.0" ]; then
	echo "Fin du script"
	rm mon_fichier.txt
	exit 0
fi

rm mon_fichier.txt

echo "Les prédictions avec les grayscales n'étaient pas suffisantes."

# Exécuter le script Python pour les Graphes d'entropie avec le chemin en argument
python3 représentation_graphe_entropie.py "$dir_path"

# Charger le modèle à partir du fichier .pkl
python3 << END
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('/home/laura/API_PRe/Modèles/')
from convnet import Convnet

Labels = ['benjamin','berbew','ceeinject','dinwod','ganelp','gepys','mira','sfone','sillyp2p','upatre','wabot','wacatac','musecador']

# Chemin complet vers le modèle .pkl
model_path = '/home/laura/API_PRe/Modèles/cnn_model_entropy.pkl'

# Charger le modèle

with open(model_path, 'rb') as file:
         cnn_model_entropy = pickle.load(file)

# Chemin complet vers le répertoire contenant les fichiers Graphes d'entropie
entropy_dir = "/home/laura/API_PRe/API_Laura/Résultats/graphe_entropie"

#On va parcourir les fichiers dans le répertoire Graphes d'entropie
for filename in os.listdir(entropy_dir):
	file_path = os.path.join(entropy_dir, filename)

	# Charger les données à partir du fichier
	image = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
	image = image.resize((128,128))
	data = np.array(image)
	
	data = data.reshape(1, 1, data.shape[0], data.shape[1])
	data = torch.tensor(data, dtype=torch.float32)
	
	# Faire des prédictions avec le modèle
	cnn_model_entropy.eval()  # Mettre le modèle en mode évaluation

	with torch.no_grad():
                # Obtenir les sorties brutes du modèle
		outputs = cnn_model_entropy(data)
		# Appliquer la fonction softmax pour obtenir les probabilités
		probabilities = F.softmax(outputs, dim=1)
		# Obtenir l'indice de la classe prédite
		idx = probabilities.argmax(axis=1).cpu().numpy()[0]
		# Afficher les probabilités et la prédiction
		print(f"La probabilité de prédiction est de : {probabilities.cpu().numpy()[0][0]}")
		if probabilities.cpu().numpy()[0][0]==1:
			print("Le Graphe d'entropie suffit.")
		print(f"Predicted class index for {filename}: {Labels[idx]}")
		
		ma_variable = str(probabilities.cpu().numpy()[0][0])
		
		with open('mon_fichier.txt', 'w') as f:
			f.write(ma_variable)

END

valeur_recuperee=$(< mon_fichier.txt)

if [ "$valeur_recuperee" == "1.0" ]; then
	echo "Fin du script"
	rm mon_fichier.txt
	exit 0
fi

rm mon_fichier.txt

echo "Les prédictions avec les graphes d'entropie n'étaient pas suffisantes."

header="Label,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30,F31,F32,F33,F34,F35,F36,F37,F38,F39,F40,F41,F42,F43,F44,F45,F46,F47,F48,F49,F50,F51,F52,F53,F54,F55,F56,F57,F58,F59,F60,F61,F62,F63,F64,F65,F66,F67,F68,F69,F70,F71,F72,F73,F74,F75,F76,F77,F78,F79,F80,F81,F82,F83,F84,F85,F86,F87,F88,F89,F90,F91,F92,F93,F94,F95,F96,F97,F98,F99,F100,F101,F102,F103,F104,F105,F106,F107,F108,F109,F110,F111,F112,F113,F114,F115,F116,F117,F118,F119"


#Les features PE_feats
cd PE_feats
touch ../Résultats/pe_feats.csv

# Si le fichier de sortie est vide ou n'existe pas, ajouter l'entête
if [ ! -s ../Résultats/pe_feats.csv ]; then
  echo "$header" > ../Résultats/pe_feats.csv
fi

for filename in "$dir_path"/*
	do
	./pefeats "$filename" >> ../Résultats/pe_feats.csv
	done
cd ..

python3 << END
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Chemin complet vers le csv contenant les features PE_feats
pe_feats_file = "/home/laura/API_PRe/API_Laura/Résultats/pe_feats.csv"

# Chemin complet vers le modèle .pkl
model_path = '/home/laura/API_PRe/Modèles/GB_model_pe_feats.pkl'

# Charger le modèle

with open(model_path, 'rb') as file:
	GB_model_pe_feats = pickle.load(file)

#Ouvrir le fichier csv et le lire
features_pe_feats = pd.read_csv(pe_feats_file)


#Récupérer les noms de colonnes
num_features = features_pe_feats.shape[1]
features_pe_feats = features_pe_feats.iloc[:, 1:]
new_column_names = [f'F{i+1}' for i in range(num_features-1)]
features_pe_feats.columns = new_column_names

print("Selon PE_feats, la classe prédite est :")
# Faire des prédictions avec le modèle chargé
predictions = GB_model_pe_feats.predict(features_pe_feats)

# Afficher les prédictions
print(predictions)

# Afficher un message avant de faire les prédictions
print("Selon PE_feats, la probabilité de prédiction est :")

# Faire des prédictions avec les probabilités
predictions_proba = GB_model_pe_feats.predict_proba(features_pe_feats)

u = max(predictions_proba[0][0],predictions_proba[0][1],predictions_proba[0][2],predictions_proba[0][3],predictions_proba[0][4],predictions_proba[0][5],predictions_proba[0][6],predictions_proba[0][7],predictions_proba[0][8],predictions_proba[0][9],predictions_proba[0][10],predictions_proba[0][11],predictions_proba[0][12],predictions_proba[0][12])
print(u)

ma_variable=str(u)
with open('mon_fichier.txt', 'w') as f:
	f.write(ma_variable)

END

valeur_recuperee=$(< mon_fichier.txt)

if [ "$valeur_recuperee" == "1.0" ]; then
	echo "Fin du script"
	rm mon_fichier.txt
	exit 0
fi

rm mon_fichier.txt

echo "Les prédictions avec les pe_feats n'étaient pas suffisantes."

#Les features Ember
cd Pack_Ember
python3 représentation_ember.py "$dir_path"
cd ..

python3 << END

import pickle
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Chemin complet vers le csv contenant les features Ember
ember_file = "/home/laura/API_PRe/API_Laura/Résultats/ember.csv"

# Chemin complet vers le modèle .pkl
model_path = '/home/laura/API_PRe/Modèles/GB_model_ember.pkl'

# Charger le modèle

with open(model_path, 'rb') as file:
	GB_model_ember = pickle.load(file)

#Ouvrir le fichier csv et le lire
features_ember = pd.read_csv(ember_file)

#Récupérer les noms de colonnes
num_features = features_ember.shape[1]
features_ember = features_ember.iloc[:, 1:]
new_column_names = [f'F{i+1}' for i in range(num_features-1)]
features_ember.columns = new_column_names

print("Selon Ember, la classe prédite est :")

# Faire des prédictions avec le modèle chargé
predictions = GB_model_ember.predict(features_ember)

# Afficher les prédictions
print(predictions)

END
