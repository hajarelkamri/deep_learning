# main.py
import os
from tensorflow.keras.models import load_model
import train_model
import webcam_predict

def main():
    # Entraînement du modèle
    print("🚀 Entraînement du modèle...")
    # Tu peux commenter ou décommenter cette ligne en fonction de si tu veux réentraîner ou non
    train_model.main()  # Décommente cette ligne si tu veux entraîner le modèle

    # Charger le modèle sauvegardé
    model = load_model("mon_model.h5")

    # Prédiction en temps réel sur la webcam
    webcam_predict.main()

if __name__ == "__main__":
    main()
