"""
Script pour pré-générer tous les graphiques ML (6 points)
Selon instructions du prof - Visualisations détaillées
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import base64
from io import BytesIO

class MLVisualizer:
    """Génère les 6 graphiques ML requis par le prof"""
    
    def __init__(self, ml_analyzer):
        self.ml_analyzer = ml_analyzer
        os.makedirs("graphs", exist_ok=True)
    
    def pregenerate_all_ml_graphs(self, patients):
        """
        Pré-génère les 6 graphiques ML et les sauvegarde:
        - Random Forest: 2 graphiques
        - XGBoost: 2 graphiques  
        - K-Means: 2 graphiques
        + BONUS: 3 graphiques EDF (Age, Income, Number of Children)
        """
        print(" Pré-génération des visualisations ML...")
        
        if not self.ml_analyzer.is_trained:
            print(" Modèles ML non entraînés - impossible de générer les graphiques")
            return False
        
        df = pd.DataFrame(patients)
        
        try:
            # 1-2. Random Forest (2 graphiques)
            print("  1/9 Random Forest - Importance features...")
            self._generate_rf_importance(df)
            
            print("  2/9 Random Forest - Matrice confusion...")
            self._generate_rf_confusion(df)
            
            # 3-4. XGBoost (2 graphiques)
            print("  3/9 XGBoost - Actual vs Predicted...")
            self._generate_xgboost_actual_vs_predicted(df)
            
            print("  4/9 XGBoost - Distribution scores...")
            self._generate_xgboost_distribution(df)
            
            # 5-6. K-Means (2 graphiques)
            print("  5/9 K-Means - Clusters 2D...")
            self._generate_kmeans_2d(df)
            
            print("  6/9 K-Means - Distribution clusters...")
            self._generate_kmeans_distribution(df)
            
            # 7-9. EDF (3 graphiques) - BONUS!
            print("  7/9 EDF - Age...")
            self._generate_edf_graph(df, "Age")
            
            print("  8/9 EDF - Income...")
            self._generate_edf_graph(df, "Income")
            
            print("  9/9 EDF - Number of Children...")
            self._generate_edf_graph(df, "Number of Children")
            
            print(" Tous les graphiques ML + EDF pré-générés!")
            return True
            
        except Exception as e:
            print(f" Erreur génération graphiques: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_rf_importance(self, df):
        """1. Random Forest - Importance des features"""
        # Préparer données
        X, y = self._prepare_classification_data(df)
        X_scaled = self.ml_analyzer.scaler.transform(X)
        
        # Obtenir importance
        importances = self.ml_analyzer.random_forest.feature_importances_
        feature_names = [
            "Age", "Income", "Nb Enfants", "Stress", "Support Social",
            "Mental Hist", "Family Hist", "Chronic", "Alcool", "Tabac",
            "Activité", "Sommeil", "Marital", "Education", "Occupation"
        ]
        
        # Trier par importance
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        
        # Graphique GÉANT 20x18 pour présentation!
        plt.figure(figsize=(20, 18))
        plt.barh(range(len(indices)), importances[indices], color='#667eea')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=20)
        plt.xlabel('Importance', fontsize=22, fontweight='bold')
        plt.title('Random Forest - Top 10 Features les Plus Importantes', fontsize=26, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig('graphs/random_forest_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_rf_confusion(self, df):
        """2. Random Forest - Matrice de confusion"""
        # Préparer données
        X, y = self._prepare_classification_data(df)
        X_scaled = self.ml_analyzer.scaler.transform(X)
        
        # Prédictions
        y_pred = self.ml_analyzer.random_forest.predict(X_scaled)
        
        # Matrice de confusion
        cm = confusion_matrix(y, y_pred)
        
        # Graphique GÉANT 20x18
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Risque Faible', 'Risque Élevé'],
                    yticklabels=['Risque Faible', 'Risque Élevé'],
                    annot_kws={'fontsize': 24})
        plt.xlabel('Prédiction', fontsize=22, fontweight='bold')
        plt.ylabel('Réel', fontsize=22, fontweight='bold')
        plt.title('Random Forest - Matrice de Confusion', fontsize=26, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        
        plt.savefig('graphs/random_forest_confusion.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_xgboost_actual_vs_predicted(self, df):
        """3. XGBoost - Actual vs Predicted (comme l'exemple du prof!)"""
        # Préparer données
        X, y = self._prepare_regression_data(df)
        X_scaled = self.ml_analyzer.scaler.transform(X)
        
        # Prédictions
        y_pred = self.ml_analyzer.xgboost.predict(X_scaled)
        
        # Limiter à 0-100
        y_pred = np.clip(y_pred, 0, 100)
        y = np.clip(y, 0, 100)
        
        # Calcul MSE
        mse = np.mean((y - y_pred) ** 2)
        
        # Graphique GÉANT 20x18 (comme l'exemple du prof!)
        plt.figure(figsize=(20, 18))
        plt.scatter(y, y_pred, alpha=0.5, s=40, color='#667eea')
        
        # Ligne diagonale parfaite
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=3, label='Prédiction Parfaite')
        
        plt.xlabel('Actual Values', fontsize=22, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=22, fontweight='bold')
        plt.title(f'Actual vs. Predicted Values\nMean Squared Error: {mse:.2f}', 
                  fontsize=26, fontweight='bold')
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('graphs/xgboost_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_xgboost_distribution(self, df):
        """4. XGBoost - Distribution des scores prédits"""
        # Préparer données
        X, y = self._prepare_regression_data(df)
        X_scaled = self.ml_analyzer.scaler.transform(X)
        
        # Prédictions
        y_pred = self.ml_analyzer.xgboost.predict(X_scaled)
        y_pred = np.clip(y_pred, 0, 100)
        
        # Graphique GÉANT 20x18
        plt.figure(figsize=(20, 18))
        plt.hist(y_pred, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        plt.axvline(y_pred.mean(), color='red', linestyle='--', linewidth=3, 
                    label=f'Moyenne: {y_pred.mean():.1f}')
        plt.xlabel('Score de Bien-être Prédit (0-100)', fontsize=22, fontweight='bold')
        plt.ylabel('Nombre de Patients', fontsize=22, fontweight='bold')
        plt.title('XGBoost - Distribution des Scores de Bien-être Prédits', 
                  fontsize=26, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('graphs/xgboost_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_kmeans_2d(self, df):
        """5. K-Means - Visualisation clusters en 2D (via PCA)"""
        # Préparer données
        X = []
        for _, patient in df.iterrows():
            fv = self.ml_analyzer.extract_features(patient.to_dict())
            X.append(fv)
        
        X = np.array(X)
        X_scaled = self.ml_analyzer.scaler.transform(X)
        
        # Prédire clusters
        clusters = self.ml_analyzer.kmeans.predict(X_scaled)
        
        # Réduction dimensionnelle avec PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Graphique GÉANT 20x18
        plt.figure(figsize=(20, 18))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                             s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=22, fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=22, fontweight='bold')
        plt.title('K-Means - Visualisation des 3 Clusters (PCA)', 
                  fontsize=26, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('graphs/kmeans_clusters_2d.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_kmeans_distribution(self, df):
        """6. K-Means - Distribution des patients par cluster"""
        # Préparer données
        X = []
        for _, patient in df.iterrows():
            fv = self.ml_analyzer.extract_features(patient.to_dict())
            X.append(fv)
        
        X = np.array(X)
        X_scaled = self.ml_analyzer.scaler.transform(X)
        
        # Prédire clusters
        clusters = self.ml_analyzer.kmeans.predict(X_scaled)
        
        # Compter patients par cluster
        unique, counts = np.unique(clusters, return_counts=True)
        
        # Graphique GÉANT 20x18
        plt.figure(figsize=(20, 18))
        colors = ['#667eea', '#764ba2', '#f093fb']
        bars = plt.bar([f'Groupe {i+1}' for i in unique], counts, color=colors, edgecolor='black')
        plt.xlabel('Cluster', fontsize=22, fontweight='bold')
        plt.ylabel('Nombre de Patients', fontsize=22, fontweight='bold')
        plt.title('K-Means - Distribution des Patients par Cluster', 
                  fontsize=26, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # Ajouter les valeurs sur les barres
        for i, (cluster, count) in enumerate(zip(unique, counts)):
            plt.text(i, count + max(counts)*0.02, str(count), 
                    ha='center', va='bottom', fontweight='bold', fontsize=20)
        
        plt.tight_layout()
        
        plt.savefig('graphs/kmeans_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _prepare_classification_data(self, df):
        """Prépare X, y pour Random Forest"""
        X = []
        y_risk = []
        
        for _, patient in df.iterrows():
            fv = self.ml_analyzer.extract_features(patient.to_dict())
            X.append(fv)
            
            stress = patient.get("Stress Level", 5)
            mental_hist = 1 if patient.get("History of Mental Illness") == "Yes" else 0
            family_hist = 1 if patient.get("Family History of Depression") == "Yes" else 0
            
            risk_score = stress + mental_hist * 3 + family_hist * 2
            y_risk.append(1 if risk_score >= 6 else 0)
        
        return np.array(X), np.array(y_risk)
    
    def _prepare_regression_data(self, df):
        """Prépare X, y pour XGBoost"""
        X = []
        y_wellness = []
        
        for _, patient in df.iterrows():
            fv = self.ml_analyzer.extract_features(patient.to_dict())
            X.append(fv)
            
            stress = patient.get("Stress Level", 5)
            social = patient.get("Social Support Rating", 5)
            physical = 1 if patient.get("Physical Activity Level") in ["High", "Moderate"] else 0
            
            score = 100 - (stress * 8) + (social * 5) + (physical * 10)
            y_wellness.append(max(0, min(100, score)))
        
        return np.array(X), np.array(y_wellness)


    def _generate_edf_graph(self, df, variable):
        """Génère le graphique EDF GÉANT pour une variable"""
        from scipy import stats
        
        # Extraire les données
        data = df[variable].dropna().values
        
        if len(data) == 0:
            print(f" Pas de données pour {variable}")
            return
        
        # Calculs statistiques
        mu = np.mean(data)
        sigma = np.std(data)
        median = np.median(data)
        
        # EDF
        data_sorted = np.sort(data)
        edf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        
        # CDF normale théorique
        cdf_normal = stats.norm.cdf(data_sorted, mu, sigma)
        
        # Graphique GÉANT 20x18
        plt.figure(figsize=(7, 5))
        plt.plot(data_sorted, edf, label="EDF (Empirique)", linewidth=3, color='#667eea')
        plt.plot(data_sorted, cdf_normal, label="CDF (Normale théorique)", 
                linewidth=3, linestyle='--', color='red')
        
        # Ligne médiane
        plt.axvline(median, color='green', linestyle=':', linewidth=3, 
                   label=f'Médiane: {median:.1f}')
        
        plt.xlabel(variable, fontsize=22, fontweight='bold')
        plt.ylabel("Probabilité cumulative", fontsize=22, fontweight='bold')
        plt.title(f"Fonction de Distribution Empirique - {variable}", 
                 fontsize=26, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Nom de fichier safe
        filename = variable.replace(" ", "_").replace("/", "_")
        plt.savefig(f'graphs/edf_{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()


def pregenerate_ml_visualizations(ml_analyzer, patients):
    """Fonction helper pour générer tous les graphiques"""
    visualizer = MLVisualizer(ml_analyzer)
    return visualizer.pregenerate_all_ml_graphs(patients)