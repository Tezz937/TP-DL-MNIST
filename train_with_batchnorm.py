import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Créer les ensembles de validation
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

# Normalisation des données
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionnement des images
x_train = x_train.reshape(54000, 784)
x_val = x_val.reshape(6000, 784)
x_test = x_test.reshape(10000, 784)

# Paramètres
EPOCHS = 10
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# Fonction pour créer un modèle SANS Batch Normalization
def create_model_without_bn():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Fonction pour créer un modèle AVEC Batch Normalization
def create_model_with_bn():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Dictionnaire des modèles à tester
models_to_test = {
    "Sans_BatchNorm": create_model_without_bn(),
    "Avec_BatchNorm": create_model_with_bn()
}

# Boucle d'entraînement pour comparer
histories = {}

for model_name, model in models_to_test.items():
    print(f"\n{'='*60}")
    print(f"Entraînement du modèle: {model_name}")
    print(f"{'='*60}\n")
    
    with mlflow.start_run(run_name=f"BatchNorm_Comparison_{model_name}"):
        # Enregistrement des paramètres
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("dropout_rate", DROPOUT_RATE)
        mlflow.log_param("has_batch_norm", "Avec" in model_name)
        
        # Compilation du modèle
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entraînement
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Sauvegarde de l'historique
        histories[model_name] = history.history
        
        # Évaluation sur le test set
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n{model_name} - Précision sur test: {test_acc:.4f}")
        
        # Enregistrement des métriques finales
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        
        # Enregistrement du modèle
        mlflow.keras.log_model(model, f"mnist-model-{model_name}")

# Visualisation des résultats
print("\n" + "="*60)
print("COMPARAISON DES RÉSULTATS")
print("="*60)

# Créer des graphiques de comparaison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Loss d'entraînement
for model_name, history in histories.items():
    axes[0, 0].plot(history['loss'], label=model_name)
axes[0, 0].set_title('Loss d\'entraînement')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Loss de validation
for model_name, history in histories.items():
    axes[0, 1].plot(history['val_loss'], label=model_name)
axes[0, 1].set_title('Loss de validation')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Accuracy d'entraînement
for model_name, history in histories.items():
    axes[1, 0].plot(history['accuracy'], label=model_name)
axes[1, 0].set_title('Accuracy d\'entraînement')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Accuracy de validation
for model_name, history in histories.items():
    axes[1, 1].plot(history['val_accuracy'], label=model_name)
axes[1, 1].set_title('Accuracy de validation')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('batchnorm_comparison.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegardé: batchnorm_comparison.png")

# Résumé des résultats
print("\n" + "="*60)
print("RÉSUMÉ DES PERFORMANCES FINALES")
print("="*60)
for model_name, history in histories.items():
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    print(f"\n{model_name}:")
    print(f"  - Accuracy entraînement: {final_train_acc:.4f}")
    print(f"  - Accuracy validation: {final_val_acc:.4f}")
    print(f"  - Écart: {abs(final_train_acc - final_val_acc):.4f}")