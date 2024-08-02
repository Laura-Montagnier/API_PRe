#!/bin/bash

# Définir le chemin vers le répertoire des exécutables
dir_path="./Fichiers_Exécutables"

# Exécuter le(s) script(s) Python

#Les Grayscales
python3 représentation_grayscale.py "$dir_path"

#Les couleurs

python3 représentation_couleur.py "$dir_path"

#Les Graphes d'entropie
python3 représentation_graphe_entropie.py "$dir_path"

#Les features Ember
cd Pack_Ember
python3 représentation_ember.py "$dir_path"
cd ..

# Les features PE_feats
cd PE_feats
# Créer le fichier pe_feats.csv s'il n'existe pas
output_file="../Résultats/pe_feats.csv"
touch "$output_file"

# Vérifier si le fichier est vide et ajouter l'en-tête si nécessaire
if [ ! -s "$output_file" ]; then
    echo "name,$(seq -s, 1 119 | sed 's/[0-9]\+/F&/g')" > "$output_file"
fi

# Exécuter pefeats pour chaque fichier dans le répertoire
for filename in "$dir_path"/*
do
    ./pefeats "$filename" >> "$output_file"
done

cd ..

# Supprimer le répertoire intermédiaire
rm -r ./image_intermediaire

echo "Représentations créées."
