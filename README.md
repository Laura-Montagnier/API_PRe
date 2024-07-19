# Comment les représentations sont-elles fabriquées ?

## Ember

### Utiliser Ember

Nous utilisons le Pack Ember développé par https://github.com/elastic/ember. Il faut installer la bibliothèque grâce aux commandes :

pip install -r requirements.txt

python3 setup.py install

### Extraire les features

Dans le dossier API_Laura/Pack_Ember, on trouve le script python représentation_ember.py, écrit personnellement.
On peut l'utiliser indépendamment en donnant en argument le nom du dossier où se trouvent les fichiers exécutables dont on souhaite extraire les features.

python3 représentation_ember.py Fichiers_exécutables

## PE_feats

### Utiliser PE_feats

Il suffit de télécharger l'exécutable pe_feats. Il a été utilisé dans le papier :

Biondi, F., Enescu, M. A., Given-Wilson, T., Legay, A., Noureddine, L., & Verma, V. (2019). Effective, efficient, and robust packing detection and classification. Computers & Security, 85, 436-451.

### Extraire les features

On peut l'utiliser indépendamment en passant en argument le portable executable.

pe_feats notepad.exe

Ce qui est print est la valeur des features. (On les enregistre dans un fichier csv grâce au script global d'extraction de features.)

## Grayscale

### Créer un grayscale

Le principe est assez simple. On écrit le fichier binaire dans un csv, chaque octet dans une case. Puis, on transforme ce csv en une image de 250 pixels de large. 
Chaque pixel est en échelle de gris, avec sa couleur déterminée par la valeur de l'octet. Ensuite, on retransforme les images en une image de 250 par 250 pixels.
Cela permettra de la passer en argument à un réseau de neurones.

### Script

Le script utilisé est représentation_grayscale.py. Il prend en argument un dossier d'exécutables et transforme chacun d'entre eux en grayscales.
Il utilise un répertoire appelé images_intermédiaires qu'il faut supprimer ensuite (car il est inutile).


## Graphe d'entropie

###
