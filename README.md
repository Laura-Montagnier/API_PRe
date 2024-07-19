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

Il suffit de télécharger l'exécutable pe_feats. Il a été fourni par Charles-Henry Bertrand van Ouytsel et al.

### Extraire les features

On peut l'utiliser indépendamment en passant en argument le portable executable.

pe_feats notepad.exe

Ce qui est print est la valeur des features. (On les enregistre dans un fichier csv grâce au script global d'extraction de features.)
