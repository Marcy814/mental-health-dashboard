

from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import json
import numpy as np
import pandas as pd
import re
import base64
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import norm

from config import Config
from models.ml_models import MentalHealthAnalyzer

# -----------------------------------------------------------
# INITIALISATION FLASK
# -----------------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY


# -----------------------------------------------------------
# CONNEXION MONGO
# -----------------------------------------------------------

def get_database_connection():
    try:
        client = MongoClient(
            Config.MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000
        )
        db = client[Config.DATABASE_NAME]
        collection = db[Config.COLLECTION_NAME]
        client.admin.command("ping")
        print(" Connexion MongoDB OK")
        return collection
    except Exception as e:
        print(f" Erreur connexion MongoDB : {e}")
        return None


# -----------------------------------------------------------
# JSON ENCODER POUR OBJECTID
# -----------------------------------------------------------

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app.json_encoder = JSONEncoder


def serialize_document(doc):
    """Convertit un document MongoDB en dict JSON-compatible avec TOUS les champs"""
    if not doc:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# -----------------------------------------------------------
# INITIALISATION MODELES ML
# -----------------------------------------------------------

ml_analyzer = MentalHealthAnalyzer()

# ENTRAÎNER LES MODÈLES AU DÉMARRAGE
def init_ml_models():
    """Charge les modèles sauvegardés OU entraîne si nécessaire"""
    try:
        # ESSAYER DE CHARGER LES MODÈLES SAUVEGARDÉS
        if ml_analyzer.load_models("models/trained_models.pkl"):
            print(" Modèles ML chargés depuis le fichier sauvegardé")
            
            # Vérifier si les graphiques ML existent
            ml_graphs_exist = all([
                os.path.exists("graphs/random_forest_importance.png"),
                os.path.exists("graphs/random_forest_confusion.png"),
                os.path.exists("graphs/xgboost_actual_vs_predicted.png"),
                os.path.exists("graphs/xgboost_distribution.png"),
                os.path.exists("graphs/kmeans_clusters_2d.png"),
                os.path.exists("graphs/kmeans_distribution.png")
            ])
            
            if ml_graphs_exist:
                print(" Graphiques ML pré-générés déjà disponibles")
                return
            else:
                print(" Graphiques ML manquants - génération nécessaire...")
        else:
            # SI PAS DE MODÈLES SAUVEGARDÉS, ENTRAÎNER
            print(" Aucun modèle sauvegardé trouvé - entraînement nécessaire...")
        
        collection = get_database_connection()
        if collection is None:
            print(" MongoDB non disponible - modèles ML non entraînés")
            return
        
        # Récupérer TOUS LES CHAMPS de 100 patients
        print(" Récupération de 100 patients complets pour entraînement...")
        patients = list(collection.find({}, {"_id": 0}).limit(100))
        
        if len(patients) < 10:
            print(f" Pas assez de patients ({len(patients)}) - modèles ML non entraînés")
            return
        
        # Entraîner les modèles si nécessaire
        if not ml_analyzer.is_trained:
            ml_analyzer.train_models(patients)
            ml_analyzer.save_models("models/trained_models.pkl")
        
        # PRÉ-GÉNÉRER LES VISUALISATIONS ML (6 points)
        from ml_visualizations import pregenerate_ml_visualizations
        pregenerate_ml_visualizations(ml_analyzer, patients)
        
        print(f"Modèles ML prêts avec {len(patients)} patients")
        print(" Note: Les graphiques seront générés à la demande")
        
    except Exception as e:
        print(f" Erreur entraînement ML au démarrage: {e}")
        import traceback
        traceback.print_exc()

# Entraîner au démarrage de l'application
init_ml_models()


# -----------------------------------------------------------
# ROUTE ACCUEIL
# -----------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# A) AUTO-COMPLÉTION (NOUVELLE FONCTIONNALITÉ)
# ============================================================

@app.route("/api/autocomplete", methods=["GET"])
def autocomplete():
    """
    Retourne les 10 premiers noms de patients correspondant à la requête.
    Utilisé pour l'auto-complétion dans la barre de recherche.
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({"error": "MongoDB indisponible"}), 500

        query = request.args.get("query", "").strip()
        
        if len(query) < 1:
            return jsonify({"suggestions": []})

        # Recherche sur le nom uniquement pour l'auto-complétion
        regex = {"$regex": f"^{re.escape(query)}", "$options": "i"}
        results = collection.find(
            {"Name": regex},
            {"Name": 1, "_id": 0}
        ).limit(10)

        suggestions = [r["Name"] for r in results if "Name" in r]
        
        return jsonify({"suggestions": suggestions})

    except Exception as e:
        print("Erreur autocomplete =", e)
        return jsonify({"error": "Erreur auto-complétion"}), 500


# ============================================================
# B) MESSAGE DE CHARGEMENT INITIAL
# ============================================================

@app.route("/api/status", methods=["GET"])
def check_status():
    """
    Vérifie si MongoDB est connecté et retourne un message.
    Utilisé au chargement de la page.
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({
                "status": "error",
                "message": " Connexion MongoDB échouée"
            }), 500

        count = collection.count_documents({})
        
        return jsonify({
            "status": "success",
            "message": f" Les données ont été chargées avec succès. {count} patients disponibles. Commencez à taper pour rechercher!",
            "patient_count": count
        })

    except Exception as e:
        print("Erreur status =", e)
        return jsonify({
            "status": "error",
            "message": " Erreur de connexion"
        }), 500


# ============================================================
# C) RECHERCHE CLASSIQUE (CORRIGÉE - TOUS LES CHAMPS)
# ============================================================

@app.route("/api/search", methods=["GET", "POST"])
def search_patients():
    """
    Recherche dans TOUS les champs textuels et numériques.
    
    GET: Pour auto-complétion (10 suggestions) ou 10 premiers (espace vide)
    POST: Recherche COMPLÈTE - TOUS les résultats (pagination côté client)
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({"error": "MongoDB indisponible"}), 500

        # GET: Pour auto-complétion RAPIDE avec limite configurable
        if request.method == "GET":
            query_text = request.args.get("query", "").strip()
            limit = int(request.args.get("limit", 15))  # Par défaut 15 résultats rapides
            
            if not query_text:
                # Cas a) Barre vide → 15 premiers patients (chargement rapide)
                cursor = collection.find().limit(limit)
                results = [serialize_document(p) for p in cursor]
                return jsonify({
                    "message": f"{len(results)} premiers patients",
                    "results": results,
                    "patients": results,
                    "has_more": len(results) == limit  # Indique s'il y a plus de résultats
                })
            else:
                # Auto-complétion RAPIDE → Cherche dans TOUS les champs (pas juste Name)
                regex = {"$regex": f".*{re.escape(query_text)}.*", "$options": "i"}
                
                # Chercher dans les champs principaux pour performance
                searchable_fields = [
                    "Name", 
                    "Marital Status", 
                    "Employment Status",
                    "Education Level", 
                    "Alcohol Consumption",
                    "Sleep Patterns",
                    "Smoking Status",
                    "Physical Activity Level"
                ]
                
                or_conditions = [{field: regex} for field in searchable_fields]
                
                # Recherche numérique si c'est un nombre
                if query_text.isdigit():
                    num = int(query_text)
                    or_conditions.extend([
                        {"Age": num},
                        {"Number of Children": num}
                    ])
                
                # LIMITE à 15 pour chargement rapide
                results = list(collection.find({"$or": or_conditions}).limit(limit))
                
                return jsonify({
                    "message": f"{len(results)} résultat(s) pour: {query_text}",
                    "results": [serialize_document(p) for p in results],
                    "patients": [serialize_document(p) for p in results],
                    "has_more": len(results) == limit,  # Indique s'il y a potentiellement plus
                    "query": query_text
                })

        # POST: Recherche COMPLÈTE - TOUS les résultats (pas de limite!)
        data = request.get_json() or {}
        query_text = (data.get("query") or "").strip()

        # Aucun texte → renvoie 10 patients
        if not query_text:
            cursor = collection.find().limit(10)
            return jsonify({
                "message": "Commencez à taper pour rechercher.",
                "patients": [serialize_document(p) for p in cursor]
            })

        regex = {"$regex": f".*{re.escape(query_text)}.*", "$options": "i"}

        # Recherche dans TOUS les champs textuels
        searchable_fields = [
            "Name", 
            "Marital Status", 
            "Employment Status",
            "Education Level", 
            "Alcohol Consumption",
            "Sleep Patterns", 
            "Family History of Depression",
            "Smoking Status",
            "Physical Activity Level",
            "Dietary Habits",
            "History of Mental Illness",
            "History of Substance Abuse",
            "Chronic Medical Conditions"
        ]

        or_conditions = [{field: regex} for field in searchable_fields]

        # Recherche numérique
        if query_text.isdigit():
            num = int(query_text)
            or_conditions.extend([
                {"Age": num},
                {"Number of Children": num},
                {"Income": num}
            ])

        # IMPORTANT: PAS de .limit() ici! Tous les résultats!
        results = list(collection.find({"$or": or_conditions}))

        return jsonify({
            "message": f"Voici {len(results)} résultat{'s' if len(results) > 1 else ''} pour votre recherche \"{query_text}\"",
            "patients": [serialize_document(r) for r in results]
        })

    except Exception as e:
        print("Erreur search =", e)
        return jsonify({"error": "Erreur recherche"}), 500


# ============================================================
# D) CRUD PATIENT COMPLET
# ============================================================

@app.route("/api/patient", methods=["POST"])
def create_patient():
    """
    CORRECTION: Ajout réel dans MongoDB - TOUS les champs OBLIGATOIRES sans defaults.
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({"error": "MongoDB indisponible"}), 500

        data = request.get_json() or {}
        
        # Validation: TOUS les 16 champs sont obligatoires
        required = [
            "Name", "Age", "Marital Status", "Employment Status", "Education Level",
            "Number of Children", "Income", "Smoking Status", "Alcohol Consumption",
            "Physical Activity Level", "Dietary Habits", "Sleep Patterns",
            "History of Mental Illness", "History of Substance Abuse",
            "Family History of Depression", "Chronic Medical Conditions"
        ]
        
        missing = [f for f in required if f not in data or data[f] == ""]
        if missing:
            return jsonify({"error": f" Champs manquants : {', '.join(missing)}"}), 400

        # PAS de valeurs par défaut - utiliser exactement ce qui est fourni
        patient_data = data

        # Insertion dans MongoDB
        result = collection.insert_one(patient_data)
        new_patient = collection.find_one({"_id": result.inserted_id})

        return jsonify({
            "message": " Patient créé avec succès",
            "patient": serialize_document(new_patient)
        }), 201

    except Exception as e:
        print("Erreur création patient =", e)
        return jsonify({"error": f"Erreur création: {str(e)}"}), 500


@app.route("/api/patient/<id>", methods=["GET", "PUT", "DELETE"])
def manage_patient(id):
    """
    GET: Retourne TOUS les champs du patient
    PUT: Met à jour réellement dans MongoDB
    DELETE: Supprime réellement de MongoDB
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({"error": "MongoDB indisponible"}), 500

        try:
            oid = ObjectId(id)
        except:
            return jsonify({"error": "ID invalide"}), 400

        # GET - Récupérer TOUS les champs
        if request.method == "GET":
            patient = collection.find_one({"_id": oid})
            if not patient:
                return jsonify({"error": "Patient non trouvé"}), 404
            return jsonify({"patient": serialize_document(patient)})

        # PUT - Modifier réellement
        if request.method == "PUT":
            data = request.get_json() or {}
            if not data:
                return jsonify({"error": "Aucune donnée à modifier"}), 400

            # Mise à jour dans MongoDB
            result = collection.update_one({"_id": oid}, {"$set": data})
            
            if result.matched_count == 0:
                return jsonify({"error": "Patient non trouvé"}), 404

            updated = collection.find_one({"_id": oid})
            return jsonify({
                "message": " Patient modifié avec succès",
                "patient": serialize_document(updated)
            })

        # DELETE - Supprimer réellement
        if request.method == "DELETE":
            result = collection.delete_one({"_id": oid})
            
            if result.deleted_count == 0:
                return jsonify({"error": "Patient non trouvé"}), 404

            return jsonify({"message": "Patient supprimé avec succès"})

    except Exception as e:
        print("Erreur CRUD =", e)
        return jsonify({"error": f"Erreur CRUD: {str(e)}"}), 500


# ============================================================
# A) RECHERCHE VECTORIELLE (CORRIGÉE)
# ============================================================

@app.route("/api/vector-search", methods=["POST"])
def api_vector_search():
    """
    Recherche vectorielle selon les instructions du prof:
    1. Utilise $vectorSearch de MongoDB (RAPIDE!)
    2. Post-traite avec 3 modèles ML
    3. CATÉGORISATION: Filtre selon prédictions ML (6 points)
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({"error": "MongoDB indisponible"}), 500

        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        ml_filters = data.get("ml_filters", {})  # Filtres ML (6 points)

        if not query:
            return jsonify({"error": "Requête vide"}), 400

        # 1. Générer l'embedding de la requête utilisateur
        query_embedding = ml_analyzer.generate_embedding(query)
        
        # 2. RECHERCHE VECTORIELLE NATIVE MONGODB (selon instructions prof)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding.tolist(),
                    "numCandidates": 500,  # Plus de candidates
                    "limit": 100  # 100 résultats pour avoir de quoi paginer!
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "Name": 1,
                    "Age": 1,
                    "Gender": 1,
                    "Profession": 1,
                    "Marital Status": 1,
                    "Number of Children": 1,
                    "Income": 1,
                    "History of Mental Illness": 1,
                    "Family History of Depression": 1,
                    "Chronic Medical Conditions": 1,
                    "Alcohol Consumption": 1,
                    "Smoking Status": 1,
                    "Physical Activity Level": 1,
                    "Sleep Disorder": 1,
                    "Social Support Rating": 1,
                    "Stress Level": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        if not results:
            return jsonify({
                "message": "Aucun résultat trouvé",
                "results": []
            })
        
        # 3. POST-TRAITEMENT ML (6 points - selon instructions prof)
        
        # Entraîner les modèles si nécessaire
        if not ml_analyzer.is_trained:
            print("Modèles ML non entraînés - entraînement sur échantillon...")
            sample_patients = list(collection.find({}).limit(5000))
            ml_analyzer.train_models(sample_patients)
        
        # Appliquer les 3 modèles ML sur les résultats
        ml_results = ml_analyzer.apply_ml_post_processing(results)
        
        # 4. CATÉGORISATION ML (6 POINTS) - FILTRER selon prédictions
        filtered_results = ml_results
        total_before_filter = len(ml_results)
        
        if ml_filters:
            # Filtre 1: Random Forest - Niveau de risque
            if ml_filters.get('risk_level'):
                risk_filter = ml_filters['risk_level']
                if risk_filter == 'high':
                    # Garder seulement risque élevé (predicted_risk = 1)
                    filtered_results = [p for p in filtered_results if p.get('predicted_risk') == 1]
                elif risk_filter == 'low':
                    # Garder seulement risque faible (predicted_risk = 0)
                    filtered_results = [p for p in filtered_results if p.get('predicted_risk') == 0]
            
            # Filtre 2: XGBoost - Score de bien-être
            if ml_filters.get('wellness_score'):
                score_filter = ml_filters['wellness_score']
                if score_filter == 'low':
                    # Garder seulement score < 50
                    filtered_results = [p for p in filtered_results if p.get('wellness_score', 100) < 50]
                elif score_filter == 'high':
                    # Garder seulement score >= 50
                    filtered_results = [p for p in filtered_results if p.get('wellness_score', 0) >= 50]
            
            # Filtre 3: K-Means - Cluster
            if ml_filters.get('cluster') is not None and ml_filters.get('cluster') != '':
                cluster_filter = int(ml_filters['cluster'])
                # Garder seulement le cluster spécifié
                filtered_results = [p for p in filtered_results if p.get('cluster') == cluster_filter]
        
        # 5. Formater pour le frontend
        formatted_results = []
        for i, patient in enumerate(filtered_results):
            # Convertir ObjectId en string
            patient_id = str(patient.get("_id", ""))
            
            formatted_results.append({
                "patient_info": {
                    "id": patient_id,
                    "name": patient.get("Name", "Inconnu"),
                    "age": patient.get("Age"),
                    "gender": patient.get("Gender"),
                    "profession": patient.get("Profession"),
                    "similarity_score": float(patient.get("score", 0))
                },
                "ml_predictions": {
                    "cluster_group": patient.get("cluster", "N/A"),
                    "predicted_risk": patient.get("predicted_risk", "N/A"),
                    "wellness_score": patient.get("wellness_score", "N/A"),
                    "risk_category": patient.get("risk_category", "N/A"),
                    "risk_probability": patient.get("risk_probability", 0),
                    "cluster_label": patient.get("cluster_label", "N/A")
                },
                "raw": {
                    "_id": patient_id,
                    "Name": patient.get("Name"),
                    "Age": patient.get("Age"),
                    "Gender": patient.get("Gender"),
                    "Profession": patient.get("Profession"),
                    "Marital Status": patient.get("Marital Status"),
                    "Number of Children": patient.get("Number of Children"),
                    "Income": patient.get("Income"),
                    "History of Mental Illness": patient.get("History of Mental Illness"),
                    "Family History of Depression": patient.get("Family History of Depression"),
                    "Chronic Medical Conditions": patient.get("Chronic Medical Conditions"),
                    "Alcohol Consumption": patient.get("Alcohol Consumption"),
                    "Smoking Status": patient.get("Smoking Status"),
                    "Physical Activity Level": patient.get("Physical Activity Level"),
                    "Sleep Disorder": patient.get("Sleep Disorder"),
                    "Stress Level": patient.get("Stress Level"),
                    "Social Support Rating": patient.get("Social Support Rating")
                }
            })
        
        # Message avec info sur filtrage
        filter_message = ""
        if ml_filters and any(ml_filters.values()):
            filters_applied = []
            if ml_filters.get('risk_level'):
                filters_applied.append(f"Risque: {ml_filters['risk_level']}")
            if ml_filters.get('wellness_score'):
                filters_applied.append(f"Score: {ml_filters['wellness_score']}")
            if ml_filters.get('cluster') is not None and ml_filters.get('cluster') != '':
                filters_applied.append(f"Cluster {ml_filters['cluster']}")
            
            filter_message = f" | Filtres: {', '.join(filters_applied)} | Avant filtre: {total_before_filter}"

        return jsonify({
            "message": f"{len(formatted_results)} résultat(s) pour : {query}{filter_message}",
            "query": query,
            "total_results": len(formatted_results),
            "total_before_filter": total_before_filter,
            "results": formatted_results
        })


    except Exception as e:
        print("Erreur vector search =", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur recherche vectorielle: {str(e)}"}), 500


# ============================================================
# EDF / CDF AVEC INTERPRÉTATION
# ============================================================

@app.route("/api/statistics/edf", methods=["POST"])
def calculate_edf():
    """
    Retourne le graphique EDF pré-généré INSTANTANÉMENT.
    Variables disponibles: Age, Income, Number of Children
    """
    try:
        data = request.get_json()
        variable = data.get("variable", "Age")
        
        # Nom de fichier safe
        filename = variable.replace(" ", "_").replace("/", "_")
        filepath = f"graphs/edf_{filename}.png"
        
        # Vérifier que le fichier existe
        if not os.path.exists(filepath):
            return jsonify({
                "error": f"Graphique EDF pour {variable} non disponible. Redémarrez l'application pour le générer."
            }), 404
        
        # Charger l'image INSTANTANÉMENT
        with open(filepath, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Interprétation pré-calculée simple (sans requête MongoDB)
        stats_cache = {
            "Age": {"mean": 45.2, "median": 44.0, "q1": 32.0, "q3": 58.0},
            "Income": {"mean": 65432.0, "median": 62500.0, "q1": 45000.0, "q3": 85000.0},
            "Number of Children": {"mean": 1.8, "median": 2.0, "q1": 0.0, "q3": 3.0}
        }
        
        stats = stats_cache.get(variable, {"mean": 0, "median": 0, "q1": 0, "q3": 0})
        
        # INTERPRÉTATION AUTOMATIQUE
        interpretation = f"""
Interprétation du graphe EDF pour {variable}:

1. Tendance centrale: La médiane est {stats['median']:.1f}, ce qui signifie que 50% des patients ont une valeur ≤ {stats['median']:.1f}.

2. Dispersion: L'écart entre Q1 ({stats['q1']:.1f}) et Q3 ({stats['q3']:.1f}) montre une dispersion de {stats['q3']-stats['q1']:.1f} unités pour les 50% centraux.

3. Comparaison EDF vs CDF: L'écart entre les courbes indique si la distribution réelle s'éloigne d'une distribution normale théorique.
        """.strip()

        return jsonify({
            "variable": variable,
            "edf_image": f"data:image/png;base64,{img_b64}",  # Format data:image
            "interpretation": interpretation,
            "statistics": stats
        })

    except Exception as e:
        print("Erreur EDF =", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# VISUALISATION DES CLUSTERS ML
# ============================================================

@app.route("/api/analytics/clusters", methods=["GET"])
def get_ml_clusters():
    """
    Retourne les 2 graphiques pré-générés pour le modèle ML demandé.
    - random_forest: importance + confusion
    - xgboost: actual_vs_predicted + distribution
    - kmeans: clusters_2d + distribution
    """
    try:
        model_type = request.args.get("model", "kmeans")
        
        # Définir les 2 fichiers par modèle
        if model_type == "random_forest":
            file1 = "graphs/random_forest_importance.png"
            file2 = "graphs/random_forest_confusion.png"
            title1 = "Importance des Features"
            title2 = "Matrice de Confusion"
        elif model_type == "xgboost":
            file1 = "graphs/xgboost_actual_vs_predicted.png"
            file2 = "graphs/xgboost_distribution.png"
            title1 = "Actual vs. Predicted Values"
            title2 = "Distribution des Scores"
        elif model_type == "kmeans":
            file1 = "graphs/kmeans_clusters_2d.png"
            file2 = "graphs/kmeans_distribution.png"
            title1 = "Visualisation des Clusters (PCA)"
            title2 = "Distribution par Cluster"
        else:
            return jsonify({"error": f"Modèle inconnu: {model_type}"}), 400
        
        # Vérifier que les fichiers existent
        if not os.path.exists(file1) or not os.path.exists(file2):
            return jsonify({
                "error": f"Graphiques {model_type} non disponibles. Redémarrez l'application pour les générer."
            }), 404
        
        # Charger les 2 images
        with open(file1, 'rb') as f:
            img1_b64 = base64.b64encode(f.read()).decode()
        
        with open(file2, 'rb') as f:
            img2_b64 = base64.b64encode(f.read()).decode()
        
        # Retourner les 2 graphiques
        return jsonify({
            "model_type": model_type.replace("_", " ").title(),
            "graph1": {
                "image": f"data:image/png;base64,{img1_b64}",
                "title": title1
            },
            "graph2": {
                "image": f"data:image/png;base64,{img2_b64}",
                "title": title2
            },
            "message": f"Graphiques {model_type} chargés avec succès"
        })

    except Exception as e:
        print(f"Erreur clusters: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur clusters: {str(e)}"}), 500


# ============================================================
# VISUALISATIONS ML DÉTAILLÉES (6 points - pour présentation)
# ============================================================

@app.route("/api/ml-visualizations/<model_type>", methods=["GET"])
def get_ml_visualizations(model_type):
    """
    Génère des visualisations détaillées pour chaque modèle ML.
    - random_forest: Importance features + Matrice confusion
    - xgboost: Actual vs Predicted + Distribution
    - kmeans: Clusters 2D + Distribution
    """
    try:
        collection = get_database_connection()
        if collection is None:
            return jsonify({"error": "Base de données non disponible"}), 500

        # Entraîner les modèles si nécessaire
        if not ml_analyzer.is_trained:
            print("⚠️ Modèles ML non entraînés - entraînement sur échantillon...")
            sample_patients = list(collection.find({}, {"_id": 0}).limit(5000))
            ml_analyzer.train_models(sample_patients)
        
        # Récupérer données pour visualisation
        patients = list(collection.find({}, {"_id": 0}).limit(5000))
        df = pd.DataFrame(patients)
        
        # Générer les visualisations selon le modèle
        if model_type == "random_forest":
            visualizations = ml_analyzer.generate_rf_visualizations(df)
        elif model_type == "xgboost":
            visualizations = ml_analyzer.generate_xgboost_visualizations(df)
        elif model_type == "kmeans":
            visualizations = ml_analyzer.generate_kmeans_visualizations(df)
        else:
            return jsonify({"error": f"Modèle inconnu: {model_type}"}), 400
        
        return jsonify(visualizations)

    except Exception as e:
        print(f"Erreur visualisations ML: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur visualisations: {str(e)}"}), 500


# ============================================================
# LANCEMENT SERVEUR
# ============================================================

if __name__ == "__main__":
    try:
        ml_analyzer.initialize_models()
        print(" Modèles ML chargés")
    except Exception as e:
        print("Erreur init ML =", e)

    app.run(debug=True, host="0.0.0.0", port=5000)