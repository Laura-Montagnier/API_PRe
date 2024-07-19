#!/bin/bash

# Chemin vers le répertoire des exécutables
dir_path="/home/laura/API_PRe/API_Laura/Fichiers_exécutables/"

# Exécuter le(s) script(s) Python

#Les Grayscales
python3 représentation_grayscale.py "$dir_path"

#Les Graphes d'entropie
python3 représentation_graphe_entropie.py "$dir_path"

#Les features Ember
cd Pack_Ember
python3 représentation_ember.py "$dir_path"
cd ..

# Les features PE_feats
cd PE_feats
touch ../Résultats/pe_feats.csv
for filename in "$dir_path"/*
do
    ./pefeats "$filename" >> ../Résultats/pe_feats.csv
done
cd ..

# Supprimer le répertoire intermédiaire
rm -r ~/API_PRe/API_Laura/image_intermediaire

echo "Représentations créées."
