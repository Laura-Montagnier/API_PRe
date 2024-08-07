#!/bin/bash

# Définir les dossiers
dossier_source="./../../bodmas1_samples"
dossier_destination="Fichiers_exécutables"
script_detection="./test_pour_détection.sh"
fichier_csv="Détection.csv"

# Créer le dossier destination s'il n'existe pas
mkdir -p "$dossier_destination"

# Initialiser le fichier CSV avec l'en-tête
echo "Type,Sortie du script,Modèle responsable" > "$fichier_csv"

# Parcourir tous les fichiers dans le dossier source et ses sous-dossiers
find "$dossier_source" -type f | while read -r fichier; do
    # Obtenir le nom du fichier sans le chemin
    nom_fichier=$(basename "$fichier")

    # Obtenir le nom du sous-dossier (le premier sous-dossier parent après dossier_source)
    sous_dossier=$(dirname "$fichier" | sed "s|$dossier_source/||" | cut -d/ -f1)

    # Copier le fichier dans le dossier destination
    cp "$fichier" "$dossier_destination/"

    # Exécuter le script de détection et capturer la sortie
    sortie_script=$("$script_detection" "$dossier_destination/$nom_fichier")

    # Ajouter une entrée dans le fichier CSV
    echo "$sous_dossier,$sortie_script" >> "$fichier_csv"

    # Supprimer le fichier copié pour éviter des conflits
    rm "$dossier_destination/$nom_fichier"
done

echo "Traitement terminé. Les résultats sont dans $fichier_csv."

