from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = keras.models.load_model("mnist_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if "image" not in data:
        return jsonify({"error": "Aucune image reçue"}), 400

    # Conversion en tableau numpy
    image_data = np.array(data["image"])
    image_data = image_data.reshape(1, 784).astype("float32") / 255.0

    # Prédiction
    prediction = model.predict(image_data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return jsonify({
        "prediction": predicted_class,
        "probabilities": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
