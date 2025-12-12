 Créer un environnement virtuel (RECOMMANDÉ)
bash# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Vous devriez voir (venv) au début de votre ligne de commande.

Étape 3: Installer les dépendances
bash# Installation en UNE commande
pip install -r requirements.txt
Note: L'installation prend 5-10 minutes (télécharge ~500 MB de dépendances).
Dépendances principales installées:

Flask, PyMongo (backend)
sentence-transformers, torch (embeddings)
scikit-learn, xgboost (ML)
matplotlib, seaborn (visualisations)
pandas, numpy, scipy (calculs)


 Configuration
Étape 1: Copier le fichier .env.example
bash# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
Étape 2: Configurer MongoDB Atlas
A. Créer un cluster MongoDB Atlas (si pas déjà fait):

Aller sur mongodb.com/cloud/atlas
Créer un compte gratuit
Créer un cluster M0 (gratuit)
Créer un utilisateur (username + password)
Whitelist votre IP (ou 0.0.0.0/0 pour accès partout)

B. Récupérer l'URI de connexion:

Cliquer sur "Connect" sur votre cluster
Choisir "Connect your application"
Copier l'URI (format: mongodb+srv://...)

C. Modifier le fichier .env:
env# Remplacer <username>, <password>, et <cluster> par vos valeurs
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Exemple réel:
# MONGODB_URI=mongodb+srv://john:MyP@ssw0rd@cluster0.abc123.mongodb.net/

# Nom de la base de données
DB_NAME=Depression

# Nom de la collection
COLLECTION_NAME=depression

# Port Flask (laisser 5000)
FLASK_PORT=5000

# Mode debug (laisser True pour développement)
FLASK_DEBUG=True
 IMPORTANT: Ne JAMAIS partager votre fichier .env (il contient vos mots de passe!)
Étape 3: Vérifier la connexion MongoDB
bash# Tester la connexion
python -c "from pymongo import MongoClient; client = MongoClient('VOTRE_URI'); print('✅ Connexion réussie!'); print(f'Bases de données: {client.list_database_names()}')"
Si erreur, vérifier:

URI correcte dans .env
Username/Password corrects
IP whitelistée dans MongoDB Atlas
Connexion Internet active


 Lancement
Démarrer l'application
bash# Activer l'environnement virtuel (si pas déjà fait)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Lancer Flask
python app.py
Output attendu:
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: xxx-xxx-xxx
Accéder à l'application
Ouvrir votre navigateur et aller sur:
http://localhost:5000
ou
http://127.0.0.1:5000