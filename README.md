# PREMIER SCRIPT : représentations.sh

## Description

Ce script permet de créer différentes représentations de fichiers exécutables (EXE). Elle génère plusieurs types de features et les enregistre dans le dossier `Résultats`.

## Instructions

### 1. Placer les fichiers exécutables

Placez les fichiers exécutables dans le répertoire `Fichiers_Exécutables`.

### 2. Adapter ember 

pip install -r requirements.txt

python3 setup.py install

### 3. Installer LEAF

Installer la version 14 de Leaf.

### 3. Rendre le script exécutable

Ouvrez un terminal et rendez le script `représentations.sh` exécutable en entrant la commande suivante :

chmod +x représentations.sh

### 4. Exécuter le script

./représentations.sh

## Résultats

Les différents types de features sont présents dans le dossier "Résultats".

## Références

Pour le Pack_EMBER : https://github.com/elastic/ember

Pour PE_Feats : https://github.com/packing-box/awesome-executable-packing.

# APARTE : Comment les représentations sont-elles fabriquées ?

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

Il faut lui donner la permission pour être exécutable :

chmod +x ~/API_Laura/PE_feats/pefeats

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

### Créer le graphe d'entropie 

On calcule la fréquence de chaque valeur de byte. Ainsi, on peut calculer l'entropie de chaque byte. On trace cette entropie en fonction de la valeur du byte, ce qui donne un graphe.

### Script

Le script utilisé est représentation_graphe_entropie.py. Il prend en argument un dossier d'exécutables et trace pour chacun d'entre eux son graphe d'entropie.

# DEUXIEME SCRIPT : détection.sh

## Description

Ce script permet de déterminer si un fichier exécutable PE est un Cleanware ou un Malware.

## Insctructions

### Créer les représentations

Ce script nécessite que le script représentations.sh fonctionne. Il faut donc installer les pré-requis du premier script.

### Fonctionnement

détection.sh ne permet de classifier qu'un seul exécutable à la fois. Il faut donc placer cet exécutable dans le dossier Fichiers_exécutables et vérifier qu'il n'y en a pas d'autres.

### Le dossier Résultats

Il faut supprimer le dossier API_Laura/Résultats : "rm -r Résultats" après chaque utilisation. Sauf si vous souhaitez observer les représentations et/ou les conserver.
Néanmoins l'API ne fonctionnera bien que si le dossier est supprimé (ou vidé) avant chaque utilisation.

# TROISIEME SCRIPT : classification.sh

## Description

Ce script permet de classifier des exécutables malveillants en 14 familles différentes.

## Insctructions

### Créer les représentations

Ce script nécessite que le script représentations.sh fonctionne. Il faut donc installer les pré-requis du premier script.

### Fonctionnement

classification.sh ne permet de classifier qu'un seul exécutable à la fois. Il faut donc placer cet exécutable dans le dossier Fichiers_exécutables et vérifier qu'il n'y en a pas d'autres.

### Le dossier Résultats

Il faut supprimer le dossier API_Laura/Résultats : "rm -r Résultats" après chaque utilisation. Sauf si vous souhaitez observer les représentations et/ou les conserver.
Néanmoins l'API ne fonctionnera bien que si le dossier est supprimé (ou vidé) avant chaque utilisation.

### Rendre classification.sh exécutable

chmod +x classification.sh


