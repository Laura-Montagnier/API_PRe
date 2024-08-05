import os
import argparse
from PIL import Image
import numpy as np
import math

# Initialisation de la colormap
colormap = []
for i in range(16):
    row = []
    for j in range(16):
        R = (i * 16) % 256
        G = (j * 16) % 256
        B = ((i * 16) + (j * 16)) % 256
        row.append((R, G, B))
    colormap.append(row)

# Fonction pour convertir des données binaires en valeurs RGB
def binary_to_rgb(binary_data):
    rgb_values = []
    for byte in binary_data:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        rgb = colormap[high_nibble][low_nibble]
        rgb_values.append(rgb)
    return rgb_values

# Fonction pour créer une image à partir d'un fichier binaire
def create_image_from_binary(file_path, output_dir, width=250, target_size=(250, 250)):
    # Lire les données binaires du fichier
    with open(file_path, 'rb') as f:
        binary_data = f.read()

    # Nombre total de pixels
    total_pixels = len(binary_data)

    # Calculer la hauteur nécessaire
    height = math.ceil(total_pixels / width)

    # Convertir les données binaires en valeurs RGB
    rgb_values = binary_to_rgb(binary_data)

    # Créer une image à partir des valeurs RGB
    image = Image.new('RGB', (width, height))

    # Si le nombre de pixels RGB est inférieur à la taille de l'image, ajouter des pixels noirs
    if len(rgb_values) < width * height:
        rgb_values.extend([(0, 0, 0)] * (width * height - len(rgb_values)))

    # Mettre les données RGB dans l'image
    image.putdata(rgb_values)

    # Redimensionner l'image
    resized_image = image.resize(target_size)

    # Sauvegarder l'image redimensionnée
    output_path = os.path.join(output_dir, os.path.basename(file_path) + '.couleur.png')
    resized_image.save(output_path)
    print(f"Image saved to {output_path}")

# Fonction pour parcourir un répertoire et traiter tous les fichiers exécutables
def process_directory(directory, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir les fichiers dans le répertoire
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            create_image_from_binary(file_path, output_dir)

# Fonction principale pour gérer les arguments de la ligne de commande
def main():
    parser = argparse.ArgumentParser(description='Convert binary executables to RGB images.')
    parser.add_argument('input_directory', help='Directory containing executable files')
    args = parser.parse_args()

    output_directory="../API_Laura/Résultats/couleur"

    process_directory(args.input_directory, output_directory)

if __name__ == "__main__":
    main()

