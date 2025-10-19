# TP-DL-MNIST: Classification de Chiffres Manuscrites

Ce projet est réalisé dans le cadre de la partie "Ingénierie du Deep Learning" d'un Travaux Pratiques.

## Objectif

L'objectif de cette partie est de :
1. Construire et entraîner un premier modèle de réseau de neurones (Dense) pour la classification des chiffres manuscrits du jeu de données **MNIST**.
2. Appliquer les pratiques d'ingénierie logicielle (Versionnement Git/GitHub).
3. Intégrer les outils MLOps comme MLflow (Partie 2.2).

## État du Projet (Exercice 1)

Le modèle `mnist_model.h5` a été entraîné avec :
* **Architecture :Réseau Dense (512 neurones + Dropout + 10 neurones Softmax)
* **Jeu de données : MNIST
* **Précision (Test) : 0.9825]

## Instructions pour l'Exécution Locale

1. Cloner le dépôt :
   `git clone <URL_de_votre_dépôt>`
2. Activer l'environnement virtuel :
   `source venv/bin/activate` (Mac/Linux) ou `.\venv\Scripts\activate` (Windows)
3. Installer les dépendances :
   `pip install tensorflow numpy jupyter`
4. Lancer le notebook :
   `jupyter notebook train_model.ipynb`
