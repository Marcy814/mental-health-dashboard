# Configuration de l'application Flask et connexion MongoDB
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()


class Config:
    # Configuration MongoDB Atlas - URI complète depuis .env
    MONGO_URI = os.getenv('MONGODB_URI')

    # Vérification que l'URI est présente
    if not MONGO_URI:
        raise ValueError("MONGODB_URI non configurée dans le fichier .env")

    # Configuration base de données - CORRIGÉE
    DATABASE_NAME = 'Depression'
    COLLECTION_NAME = 'depression'

    # Clé secrète pour Flask
    SECRET_KEY = os.getenv('SECRET_KEY')

    # Vérification que la clé secrète est présente
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY non configurée dans le fichier .env")

    # Configuration Machine Learning
    MODEL_PATH = 'models/'
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', '384'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))

    # Paramètres statistiques
    EDF_BINS = 50
    CONFIDENCE_LEVEL = 0.95

    # Paramètres d'application
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('PORT', '5000'))
    HOST = os.getenv('HOST', '0.0.0.0')