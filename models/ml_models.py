# =========================================
# ML MODELS FOR MENTAL WELLNESS DASHBOARD
#
# =========================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sentence_transformers import SentenceTransformer
import xgboost as xgb  

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import base64
from io import BytesIO
import pickle
import os


class MentalHealthAnalyzer:
    def __init__(self):
        self.embedding_model = None
        self.scaler = StandardScaler()

        # Encoders pour les variables catégorielles
        self.label_encoders = {}

        # Modèles ML distincts: K-Means, Random Forest, XGBoost
        # OPTIMISÉ: Moins d'arbres = plus rapide
        self.random_forest = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        self.xgboost = xgb.XGBRegressor(n_estimators=50, random_state=42, max_depth=5, verbosity=0)
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

        self.is_trained = False

    # -------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------
    def initialize_models(self):
        """Charge le modèle d'embedding et initialise les modèles ML"""
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Modèle d'embedding chargé")
            print(" Tous les modèles ML initialisés")
        except Exception as e:
            print(" ERREUR init ML:", e)
            raise

    # -------------------------------------------------------------
    # EMBEDDING DU TEXTE
    # -------------------------------------------------------------
    def generate_embedding(self, text):
        if self.embedding_model is None:
            self.initialize_models()
        return self.embedding_model.encode([text])[0]

    # -------------------------------------------------------------
    # VECTOR SEARCH CORRIGÉ
    # -------------------------------------------------------------
    def find_similar_patients(self, query_embedding, mongo_patients, top_k=10):
        """
        CORRECTION: Prend maintenant 2 arguments comme attendu.
        mongo_patients = liste brute venant de MongoDB
        query_embedding = vecteur de la requête utilisateur
        """

        enriched_patients = []

        for p in mongo_patients:
            desc = self.build_description(p)
            emb = self.embedding_model.encode([desc])[0]

            # Similarité cosinus
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10
            )

            enriched_patients.append(
                {
                    "patient_id": str(p.get("_id", "")),
                    "similarity_score": float(sim),
                    "name": p.get("Name", "Inconnu"),
                    "age": p.get("Age", None),
                    "risk_level": "Inconnu",
                    "raw": p,
                }
            )

        enriched_patients = sorted(
            enriched_patients, key=lambda x: x["similarity_score"], reverse=True
        )

        return enriched_patients[:top_k]

    # -------------------------------------------------------------
    # DESCRIPTION TEXTUELLE POUR EMBEDDING
    # -------------------------------------------------------------
    def build_description(self, p):
        """Construit un texte cohérent pour l'embedding"""
        desc = f"""
        Name: {p.get('Name')}
        Age: {p.get('Age')}
        Marital Status: {p.get('Marital Status')}
        Education Level: {p.get('Education Level')}
        Employment Status: {p.get('Employment Status')}
        Smoking: {p.get('Smoking Status')}
        Alcohol: {p.get('Alcohol Consumption')}
        Sleep: {p.get('Sleep Patterns')}
        Physical Activity: {p.get('Physical Activity Level')}
        Income: {p.get('Income')}
        History Mental Illness: {p.get('History of Mental Illness')}
        History Substance Abuse: {p.get('History of Substance Abuse')}
        Family Depression: {p.get('Family History of Depression')}
        Chronic Conditions: {p.get('Chronic Medical Conditions')}
        """
        return desc

    # -------------------------------------------------------------
    # EXTRACTION DES FEATURES NUMÉRIQUES
    # -------------------------------------------------------------
    def extract_features(self, p):
        """Sélectionne les features utilisées par les modèles ML"""

        vector = []

        # Variables numériques
        vector.append(p.get("Age", 30))
        vector.append(p.get("Number of Children", 0))
        vector.append(p.get("Income", 50000) / 10000)

        # Variables catégorielles
        categorical = [
            "Marital Status",
            "Education Level",
            "Employment Status",
            "Smoking Status",
            "Physical Activity Level",
            "Alcohol Consumption",
            "Dietary Habits",
            "Sleep Patterns",
            "History of Mental Illness",
            "History of Substance Abuse",
            "Family History of Depression",
            "Chronic Medical Conditions",
        ]

        for var in categorical:
            val = p.get(var, "Unknown")

            if var not in self.label_encoders:
                self.label_encoders[var] = LabelEncoder()
                # Fit avec toutes les valeurs possibles
                possible_values = ["Unknown", "Low", "Medium", "High", "Yes", "No", 
                                 "Single", "Married", "Divorced", "Widowed",
                                 "Employed", "Unemployed", "Student", "Retired",
                                 "High School", "Bachelor's Degree", "Master's Degree", "PhD",
                                 "Poor", "Fair", "Good", "Excellent",
                                 "Healthy", "Unhealthy", "Sedentary", "Moderate", "Active",
                                 "Non-smoker", "Former", "Current"]
                self.label_encoders[var].fit(possible_values)

            try:
                encoded = self.label_encoders[var].transform([val])[0]
            except:
                encoded = 0

            vector.append(encoded)

        return np.array(vector)

    # -------------------------------------------------------------
    # ENTRAINEMENT DES MODÈLES SUR TES DONNÉES MONGO
    # -------------------------------------------------------------
    def train_models(self, mongo_patients):
        """
        Entraîne les 3 modèles ML distincts:
        1. Random Forest: Classification risque élevé/faible
        2. XGBoost: Prédiction score de bien-être mental
        3. K-Means: Clustering des patients
        """
        print(f"Entraînement des modèles ML avec {len(mongo_patients)} patients...")

        print("   Extraction des features...")
        X = []
        y_risk = []  # Pour Random Forest (classification binaire)
        y_wellness_score = []  # Pour XGBoost

        for i, p in enumerate(mongo_patients):
            if i % 20 == 0:  # Progression tous les 20 patients
                print(f"    Progression: {i}/{len(mongo_patients)} patients...")
            fv = self.extract_features(p)
            X.append(fv)

            # LABEL 1: Risque élevé = combinaison de facteurs
            age = p.get("Age", 30)
            mental_illness = p.get("History of Mental Illness", "No")
            family_depression = p.get("Family History of Depression", "No")
            substance_abuse = p.get("History of Substance Abuse", "No")
            
            # Risque élevé si: âge > 45 OU historique maladie OU famille dépression
            high_risk = (age > 45) or (mental_illness == "Yes") or (family_depression == "Yes") or (substance_abuse == "Yes")
            y_risk.append(1 if high_risk else 0)

            # LABEL 2: Score de bien-être mental (0-100)
            # Score basé sur plusieurs facteurs positifs/négatifs
            score = 50  # Base

            # Facteurs positifs
            if p.get("Physical Activity Level") in ["Moderate", "Active"]:
                score += 10
            if p.get("Sleep Patterns") in ["Good", "Excellent"]:
                score += 15
            if p.get("Employment Status") == "Employed":
                score += 10
            if p.get("Dietary Habits") == "Healthy":
                score += 10

            # Facteurs négatifs
            if p.get("Smoking Status") in ["Former", "Current"]:
                score -= 10
            if p.get("Alcohol Consumption") in ["High", "Medium"]:
                score -= 10
            if mental_illness == "Yes":
                score -= 20
            if substance_abuse == "Yes":
                score -= 15

            y_wellness_score.append(max(0, min(100, score)))

        X = np.array(X)
        y_risk = np.array(y_risk)
        y_wellness_score = np.array(y_wellness_score)
        
        print(f"   Features extraites: {X.shape[0]} patients, {X.shape[1]} caractéristiques")

        # Feature scaling
        print("   Normalisation des features...")
        X_scaled = self.scaler.fit_transform(X)
        print("   Features normalisées")

        # Entraînement des 3 modèles
        print("  1/3 Random Forest (50 arbres)...")
        self.random_forest.fit(X_scaled, y_risk)
        print("   Random Forest entraîné")
        
        print("  2/3 XGBoost (50 arbres)...")
        self.xgboost.fit(X_scaled, y_wellness_score)
        print("   XGBoost entraîné")
        
        print("  3/3 K-Means (3 clusters)...")
        self.kmeans.fit(X_scaled)
        print("   K-Means entraîné")

        self.is_trained = True
        print("Modèles ML entraînés avec succès!")

    def save_models(self, filepath="models/trained_models.pkl"):
        """Sauvegarde les modèles entraînés dans un fichier"""
        if not self.is_trained:
            print(" Modèles non entraînés - impossible de sauvegarder")
            return False
        
        try:
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Sauvegarder tous les modèles et le scaler
            models_data = {
                'random_forest': self.random_forest,
                'xgboost': self.xgboost,
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'is_trained': True
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(models_data, f)
            
            print(f" Modèles sauvegardés: {filepath}")
            return True
            
        except Exception as e:
            print(f" Erreur sauvegarde modèles: {e}")
            return False
    
    def load_models(self, filepath="models/trained_models.pkl"):
        """Charge les modèles depuis un fichier"""
        if not os.path.exists(filepath):
            print(f" Fichier modèles introuvable: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                models_data = pickle.load(f)
            
            self.random_forest = models_data['random_forest']
            self.xgboost = models_data['xgboost']
            self.kmeans = models_data['kmeans']
            self.scaler = models_data['scaler']
            self.is_trained = models_data['is_trained']
            
            print(f" Modèles chargés depuis: {filepath}")
            return True
            
        except Exception as e:
            print(f" Erreur chargement modèles: {e}")
            return False

    def pregenerate_all_graphs(self, mongo_patients):
        """Pré-génère TOUS les graphiques (EDF + ML) et les sauvegarde"""
        print(" Pré-génération de tous les graphiques...")
        
        # Créer le dossier graphs
        os.makedirs("graphs", exist_ok=True)
        
        # Convertir en DataFrame
        df = pd.DataFrame(mongo_patients)
        
        # 1. Pré-générer les 3 EDF
        print("  1/6 EDF pour Age...")
        self._pregenerate_edf(df, "Age")
        
        print("  2/6 EDF pour Income...")
        self._pregenerate_edf(df, "Income")
        
        print("  3/6 EDF pour Number of Children...")
        self._pregenerate_edf(df, "Number of Children")
        
        # 2. Pré-générer les 3 graphiques ML
        print("  4/6 K-Means clustering...")
        result = self._generate_kmeans_plot(df)
        if "image" in result:
            self._save_graph_from_base64(result["image"], "graphs/ml_kmeans.png")
        
        print("  5/6 Random Forest feature importance...")
        result = self._generate_rf_plot(df)
        if "image" in result:
            self._save_graph_from_base64(result["image"], "graphs/ml_random_forest.png")
        
        print("  6/6 XGBoost distribution...")
        result = self._generate_xgboost_plot(df)
        if "image" in result:
            self._save_graph_from_base64(result["image"], "graphs/ml_xgboost.png")
        
        print(" Tous les graphiques pré-générés et sauvegardés!")
    
    def _pregenerate_edf(self, df, variable):
        """Génère et sauvegarde un graphique EDF"""
        if variable not in df.columns:
            print(f"   Variable {variable} non trouvée")
            return
        
        values = df[variable].dropna()
        values = values[values.apply(lambda x: isinstance(x, (int, float)))]
        
        if len(values) < 2:
            print(f"   Pas assez de données pour {variable}")
            return
        
        values = np.sort(values)
        n = len(values)
        edf = np.arange(1, n + 1) / n
        
        mean = np.mean(values)
        std = np.std(values)
        cdf = stats.norm.cdf(values, mean, std)
        
        plt.figure(figsize=(7, 5))
        plt.plot(values, edf, label="EDF (Empirique)", marker="o", markersize=3, linewidth=2)
        plt.plot(values, cdf, label="CDF (Normale théorique)", linestyle="--", linewidth=2)
        plt.legend(loc="best")
        plt.title(f"Fonction de Distribution — {variable}")
        plt.xlabel(variable)
        plt.ylabel("Probabilité cumulative")
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder directement en fichier
        filepath = f"graphs/edf_{variable.replace(' ', '_')}.png"
        plt.savefig(filepath, format="png", dpi=120, bbox_inches="tight")
        plt.close()
    
    def _save_graph_from_base64(self, base64_string, filepath):
        """Sauvegarde un graphique depuis une chaîne base64"""
        # Extraire les données base64 (retirer le préfixe data:image/png;base64,)
        if "base64," in base64_string:
            base64_data = base64_string.split("base64,")[1]
        else:
            base64_data = base64_string
        
        # Décoder et sauvegarder
        img_data = base64.b64decode(base64_data)
        with open(filepath, 'wb') as f:
            f.write(img_data)

    # -------------------------------------------------------------
    # POST-TRAITEMENT ML (Selon instructions prof - 6 points)
    # -------------------------------------------------------------
    def apply_ml_post_processing(self, search_results):
        """
        Applique les 3 modèles ML sur les résultats de recherche vectorielle:
        1. Random Forest: Classification risque
        2. XGBoost: Prédiction score bien-être
        3. K-Means: Attribution cluster
        
        Retourne les résultats enrichis avec les prédictions ML
        """
        if not self.is_trained:
            print(" Modèles ML non entraînés")
            return search_results
        
        enriched_results = []
        
        for patient in search_results:
            try:
                # Extraire les features du patient
                features = self.extract_features(patient)
                features_scaled = self.scaler.transform([features])
                
                # 1. RANDOM FOREST: Prédire la catégorie de risque
                risk_prediction = self.random_forest.predict(features_scaled)[0]
                risk_proba = self.random_forest.predict_proba(features_scaled)[0]
                risk_category = "Risque Élevé" if risk_prediction == 1 else "Risque Faible"
                
                # 2. XGBOOST: Prédire le score de bien-être (0-100)
                wellness_score = self.xgboost.predict(features_scaled)[0]
                wellness_score = max(0, min(100, float(wellness_score)))  # Limiter à 0-100
                
                # 3. K-MEANS: Attribution au cluster
                cluster = self.kmeans.predict(features_scaled)[0]
                cluster_label = f"Groupe {int(cluster) + 1}"
                
                # Enrichir le patient avec les prédictions
                # IMPORTANT: Convertir TOUS les types NumPy en types Python natifs
                patient["predicted_risk"] = int(risk_prediction)
                patient["risk_category"] = risk_category
                patient["risk_probability"] = float(risk_proba[1])  # Probabilité risque élevé
                patient["wellness_score"] = round(float(wellness_score), 1)
                patient["cluster"] = int(cluster)
                patient["cluster_label"] = cluster_label
                
                enriched_results.append(patient)
                
            except Exception as e:
                print(f"Erreur post-traitement ML pour patient: {e}")
                import traceback
                traceback.print_exc()
                # Ajouter quand même le patient sans prédictions
                patient["predicted_risk"] = "N/A"
                patient["risk_category"] = "N/A"
                patient["wellness_score"] = "N/A"
                patient["cluster"] = "N/A"
                patient["cluster_label"] = "N/A"
                enriched_results.append(patient)
        
        return enriched_results

    # -------------------------------------------------------------
    # APPLICATION DES MODÈLES AUX PATIENTS SIMILAIRES
    # -------------------------------------------------------------
    def apply_ml_models(self, patients):
        results = []

        for p in patients:
            raw = p["raw"]

            fv = self.extract_features(raw)
            fv_scaled = self.scaler.transform([fv])

            # Prédictions des 3 modèles
            pred_rf = int(self.random_forest.predict(fv_scaled)[0])
            pred_wellness = float(self.xgboost.predict(fv_scaled)[0])
            cluster = int(self.kmeans.predict(fv_scaled)[0])

            # Niveau de risque
            if pred_rf == 1:
                risk_level = "Élevé"
            else:
                risk_level = "Modéré"

            results.append(
                {
                    "patient_info": {
                        "id": p["patient_id"],
                        "name": p["name"],
                        "age": p["age"],
                        "similarity_score": p["similarity_score"],
                        "risk_level": risk_level,
                    },
                    "ml_predictions": {
                        "cluster_group": cluster,
                        "risk_category_rf": pred_rf,
                        "depression_probability": round(pred_wellness / 100, 3),
                        "wellness_score": round(pred_wellness, 1)
                    },
                }
            )

        return results

    # -------------------------------------------------------------
    # GÉNÉRATION DE 3 GRAPHES DISTINCTS
    # -------------------------------------------------------------
    def generate_cluster_plot(self, df, model_type="kmeans"):
        """
        Génère 3 types de graphes DISTINCTS selon le modèle:
        - kmeans: Scatter plot des clusters
        - random_forest: Feature importance
        - xgboost: Distribution des prédictions
        """
        try:
            if model_type == "kmeans":
                return self._generate_kmeans_plot(df)
            elif model_type == "random_forest":
                return self._generate_rf_plot(df)
            elif model_type == "xgboost":
                return self._generate_xgboost_plot(df)
            else:
                return {"error": "Type de modèle inconnu"}

        except Exception as e:
            print(f"Erreur generate_cluster_plot: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _generate_kmeans_plot(self, df):
        """Graphe 1: K-Means clustering"""
        cols = []
        for col in ["Age", "Income", "Number of Children"]:
            if col in df.columns:
                cols.append(col)

        if len(cols) < 2:
            return {"error": "Colonnes numériques insuffisantes"}

        data = df[cols].dropna().astype(float)
        if data.empty:
            return {"error": "Données vides"}

        X = self.scaler.fit_transform(data.values)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        plt.figure(figsize=(8, 6))
        
        scatter = plt.scatter(
            data[cols[0]],
            data[cols[1]],
            c=labels,
            cmap='viridis',
            s=60,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Ajouter les centres des clusters
        centers = kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers)
        plt.scatter(
            centers_original[:, 0],
            centers_original[:, 1],
            c='red',
            s=300,
            marker='*',
            edgecolor='black',
            linewidth=2,
            label='Centres des clusters'
        )

        plt.title("K-Means Clustering - Segmentation des Patients", fontsize=14, fontweight='bold')
        plt.xlabel(cols[0], fontsize=12)
        plt.ylabel(cols[1], fontsize=12)
        plt.colorbar(scatter, label='Cluster ID')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        return {
            "image": f"data:image/png;base64,{img_b64}",
            "model_type": "K-Means",
            "n_points": int(len(data)),
            "x_axis": cols[0],
            "y_axis": cols[1],
        }

    def _generate_rf_plot(self, df):
        """Graphe 2: Random Forest - Feature Importance"""
        if not self.is_trained:
            return {"error": "Modèles non entraînés"}

        # Features utilisées (simplifié pour visualisation)
        feature_names = [
            "Age", "Nb Enfants", "Revenu",
            "Statut Marital", "Éducation", "Emploi",
            "Tabac", "Activité", "Alcool", "Alimentation",
            "Sommeil", "Maladie Mental", "Abus Substance",
            "Famille Dépression", "Maladie Chronique"
        ]

        importances = self.random_forest.feature_importances_

        # Trier par importance
        indices = np.argsort(importances)[::-1][:10]  # Top 10

        plt.figure(figsize=(10, 6))
        plt.barh(
            range(len(indices)),
            importances[indices],
            color=plt.cm.viridis(np.linspace(0, 1, len(indices)))
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance", fontsize=12)
        plt.title("Random Forest - Importance des Variables", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        return {
            "image": f"data:image/png;base64,{img_b64}",
            "model_type": "Random Forest",
            "n_points": len(importances),
            "x_axis": "Importance",
            "y_axis": "Features"
        }

    def _generate_xgboost_plot(self, df):
        """Graphe 3: XGBoost - Distribution des Scores"""
        if not self.is_trained:
            return {"error": "Modèles non entraînés"}

        cols = []
        for col in ["Age", "Income", "Number of Children"]:
            if col in df.columns:
                cols.append(col)

        if len(cols) < 2:
            return {"error": "Colonnes numériques insuffisantes"}

        data = df[cols].dropna().astype(float)
        if data.empty:
            return {"error": "Données vides"}

        # Prédire les scores de bien-être pour tous les patients
        features = []
        for _, row in data.iterrows():
            mock_patient = {
                "Age": row.get("Age", 30),
                "Number of Children": row.get("Number of Children", 0),
                "Income": row.get("Income", 50000),
                "Marital Status": "Unknown",
                "Education Level": "Unknown",
                "Employment Status": "Unknown",
                "Smoking Status": "Unknown",
                "Physical Activity Level": "Unknown",
                "Alcohol Consumption": "Unknown",
                "Dietary Habits": "Unknown",
                "Sleep Patterns": "Unknown",
                "History of Mental Illness": "No",
                "History of Substance Abuse": "No",
                "Family History of Depression": "No",
                "Chronic Medical Conditions": "No"
            }
            fv = self.extract_features(mock_patient)
            features.append(fv)

        features = np.array(features)
        features_scaled = self.scaler.transform(features)
        wellness_scores = self.xgboost.predict(features_scaled)

        plt.figure(figsize=(10, 6))
        
        # Histogramme
        plt.hist(wellness_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(wellness_scores), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(wellness_scores):.1f}')
        plt.axvline(np.median(wellness_scores), color='green', linestyle='--', linewidth=2, label=f'Médiane: {np.median(wellness_scores):.1f}')
        
        plt.xlabel("Score de Bien-être Mental (0-100)", fontsize=12)
        plt.ylabel("Nombre de Patients", fontsize=12)
        plt.title("XGBoost - Distribution des Scores de Bien-être", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        return {
            "image": f"data:image/png;base64,{img_b64}",
            "model_type": "XGBoost",
            "n_points": len(wellness_scores),
            "x_axis": "Score de Bien-être",
            "y_axis": "Fréquence"
        }

    # -------------------------------------------------------------
    # EDF + CDF + PLOT
    # -------------------------------------------------------------
    def calculate_edf(self, data, variable):
        data = np.array(data)
        data = data[np.isfinite(data)]
        data.sort()

        n = len(data)
        edf = np.arange(1, n + 1) / n

        mu = np.mean(data)
        sigma = np.std(data)
        cdf = stats.norm.cdf(data, mu, sigma)

        # Plot EDF + CDF
        plt.figure(figsize=(8, 5))
        plt.step(data, edf, where="post", label="EDF", linewidth=2)
        plt.plot(data, cdf, label="CDF normale", linestyle="--", linewidth=2)
        plt.title(f"Fonction de Distribution Empirique - {variable}")
        plt.xlabel(variable)
        plt.ylabel("Probabilité cumulative")
        plt.legend()
        plt.grid(True, alpha=0.3)

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        img = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "plot_image": "data:image/png;base64," + img,
            "statistics": {
                "mean": float(mu),
                "std": float(sigma),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            }
        }