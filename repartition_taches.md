# RÉPARTITION DES TÂCHES - MENTAL WELLNESS DASHBOARD

**Cours:** SDD 1003 - Bases de données  
**Date de remise:** 11 décembre 2025  
**Membres du groupe:**
- **Membre 1:** Djiaha Kouega Marcy Audrey DJIM66300500
- **Membre 2:** SOUFO DJUNE MIRIAM SOUM67290200

---


## MEMBRE 1:DJIAHA KOUEGA Marcy Audrey - BACKEND + ML + BASE DE DONNÉES

### 1. CONFIGURATION ET INFRASTRUCTURE (10% du projet)
**Responsabilité:** Setup initial du projet

**Tâches réalisées:**
- Configuration MongoDB Atlas (création cluster, database, collection)
- Création et configuration du fichier `config.py`
- Installation et test de toutes les dépendances Python
- Création du fichier `requirements.txt` optimisé
- Setup du modèle sentence-transformers (all-MiniLM-L6-v2)

**Fichiers:**
- `config.py`
- `requirements.txt`
- `.env.example`

**Concepts à maîtriser pour la présentation:**
- Architecture MongoDB Atlas
- Variables d'environnement
- Gestion des dépendances Python
- Connexion à MongoDB avec PyMongo

---

### 2. EMBEDDINGS + RECHERCHE VECTORIELLE (3 points = 15%)
**Responsabilité:** Implémentation complète de la recherche sémantique

**Tâches réalisées:**
- Script de génération des 41,236 embeddings (384 dimensions)
- Création de l'index vectoriel dans MongoDB Atlas
- Implémentation de la route `/api/vector-search` (app.py lignes 489-600)
- Pipeline d'agrégation MongoDB avec `$vectorSearch`
- Calcul de la similarité cosinus

**Code implémenté:**
```python
# app.py - ligne 489-600
@app.route("/api/vector-search", methods=["POST"])
def api_vector_search():
    # Création embedding de la query
    query_embedding = model.encode([query])[0].tolist()
    
    # Pipeline $vectorSearch
    pipeline = [{
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 200,
            "limit": 100
        }
    }]
    
    results = list(collection.aggregate(pipeline))
    # ...
```



---

### 3. POST-TRAITEMENT ML - CATÉGORISATION (6 points = 30%)
**Responsabilité:** Application des 3 algorithmes ML et filtrage

**Tâches réalisées:**
-  Entraînement des 3 modèles ML (Random Forest, XGBoost, K-Means)
- Sauvegarde des modèles (fichiers .pkl)
- Classe `MLAnalyzer` dans `app.py` (lignes 100-250)
- Méthode `apply_ml_post_processing()` - application des 3 modèles
- **Filtrage en cascade** dans `api_vector_search()` (lignes 549-578)
  - Filtre Random Forest (risk_level)
  - Filtre XGBoost (wellness_score)
  - Filtre K-Means (cluster)

**Code implémenté - Filtrage (6 POINTS!):**
```python
# app.py - lignes 549-578
# FILTRE 1: Random Forest
if ml_filters.get('risk_level') == 'high':
    filtered_results = [p for p in filtered_results 
                       if p.get('predicted_risk') == 1]
    filter_messages.append("Risque: Élevé")

# FILTRE 2: XGBoost
if ml_filters.get('wellness_score') == 'low':
    filtered_results = [p for p in filtered_results 
                       if p.get('wellness_score', 100) < 50]
    filter_messages.append("Score: < 50")

# FILTRE 3: K-Means
if ml_filters.get('cluster') is not None:
    cluster_filter = int(ml_filters['cluster'])
    filtered_results = [p for p in filtered_results 
                       if p.get('cluster') == cluster_filter]
    filter_messages.append(f"Cluster: {cluster_filter}")
```

**Concepts à maîtriser pour la présentation:**
- **Random Forest:** Comment fonctionne? (100 arbres de décision votent)
- **XGBoost:** Différence avec Random Forest? (Gradient Boosting)
- **K-Means:** Comment trouve-t-il les clusters? (Minimiser distance intra-cluster)
- **Filtrage en cascade:** 100 → 45 → 22 → 12 patients
- **Joblib:** Pourquoi sauvegarder les modèles?

**Résultats à expliquer:**
- Random Forest: Accuracy ~67%, Rappel 100% (AUCUN patient manqué!)
- XGBoost: MSE = 1986 (erreur moyenne ±44 points)
- K-Means: 3 clusters équilibrés (34-34-32)
- Filtrage: De 100 à 12 patients ultra-pertinents

---

### 4. VISUALISATIONS ML (6 points = 30%)
**Responsabilité:** Génération et affichage des graphiques

**Tâches réalisées:**
-  Scripts Python pour générer les 6 graphiques (20x18 pouces)
  - Random Forest: feature_importance.png + confusion.png
  - XGBoost: actual_vs_predicted.png + distribution.png
  - K-Means: clusters_2d.png + distribution.png
-  Route `/api/analytics/clusters` (app.py lignes 650-720)
-  Fonction `load_image_as_base64()` (conversion PNG → base64)

**Code implémenté:**
```python
# app.py - lignes 650-720
@app.route("/api/analytics/clusters", methods=["GET"])
def get_cluster_visualization():
    model_type = request.args.get("model", "kmeans").lower()
    
    # Sélectionner les bons fichiers
    if model_type == "kmeans":
        graph1_path = "ml_visualizations/kmeans_clusters_2d.png"
        graph2_path = "ml_visualizations/kmeans_distribution.png"
    # ...
    
    # Charger et convertir en base64
    graph1_base64 = load_image_as_base64(graph1_path)
    graph2_base64 = load_image_as_base64(graph2_path)
    
    return jsonify({
        "graph1": {"image": f"data:image/png;base64,{graph1_base64}"},
        "graph2": {"image": f"data:image/png;base64,{graph2_base64}"}
    })
```

**Concepts à maîtriser pour la présentation:**
- **Base64:** Pourquoi encoder les images? (Inclusion dans JSON)
- **Data URL:** Format `data:image/png;base64,...`
- **Matplotlib/Seaborn:** Création des graphiques
- **Feature importance:** Interprétation (Age = 25% le plus important)
- **Matrice de confusion:** TP, TN, FP, FN
- **PCA:** Réduction 13D → 2D (27.7% variance visible)

**Résultats à expliquer:**
- Random Forest: Age = variable dominante (25%)
- Matrice confusion: Rappel 100% = zéro faux négatif!
- XGBoost: Distribution normale centrée sur 47
- K-Means: 3 clusters séparés, répartition 34-34-32

---

### 5. STATISTIQUES + EDF (5%)
**Responsabilité:** Analyse statistique des variables

**Tâches réalisées:**
-  Route `/api/statistics/edf` (app.py lignes 750-850)
-  Calcul EDF pour Age, Income, Number of Children
- Comparaison avec distribution normale théorique
-  Génération graphiques avec Matplotlib

**Concepts à maîtriser pour la présentation:**
- **EDF:** Fonction de Distribution Empirique - qu'est-ce que c'est?
- **Médiane vs Moyenne:** Différences et interprétation
- **Distribution normale:** Caractéristiques (symétrique, courbe en cloche)
- **Q1, Q3:** Premier et troisième quartiles

**Résultats à expliquer:**
- Age: Distribution normale parfaite, médiane 48.5 ans
- Income: Asymétrique (log-normale), médiane 30k$, 70% < 50k$
- Children: Discrète, pic à 2 enfants (23%), 35% sans enfants

---

## MEMBRE 2: SOUFO DJUNE MIRIAM SOUM67290200- FRONTEND + AUTO-COMPLÉTION + UX

### 1. AUTO-COMPLÉTION (5 points = 25%)
**Responsabilité:** Implémentation complète de l'auto-complétion

**Tâches réalisées:**
-  Route backend `/api/autocomplete` (app.py lignes 158-185)
- Fonction JavaScript `setupAutocomplete()` (main.js lignes 137-246)
-  Fonction JavaScript `displayAutocompleteSuggestions()` (main.js lignes 234-281)
-  Gestion des événements: focus, input, click
-  Debounce de 150ms pour optimiser

**Code implémenté - Backend:**
```python
# app.py - lignes 158-185
@app.route("/api/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("query", "").strip()
    
    # Regex pour chercher au début du nom
    regex = {"$regex": f"^{re.escape(query)}", "$options": "i"}
    
    results = collection.find(
        {"Name": regex},
        {"Name": 1, "_id": 0}
    ).limit(10)
    
    suggestions = [r["Name"] for r in results if "Name" in r]
    return jsonify({"suggestions": suggestions})
```

**Code implémenté - Frontend:**
```javascript
// main.js - lignes 137-246
setupAutocomplete() {
    const searchInput = $('#searchInput');
    
    // CAS A: Focus sur barre vide → 10 premiers
    searchInput.on('focus', async function() {
        if (query.length === 0) {
            const response = await $.ajax({
                url: `/api/search?query=`,
                method: "GET"
            });
            self.displayAutocompleteSuggestions(
                response.results.slice(0, 10),
                "📋 10 premiers patients:"
            );
        }
    });
    
    // CAS B: Auto-complétion pendant frappe
    searchInput.on('input', debounce(async function() {
        const response = await $.ajax({
            url: `/api/autocomplete?query=${query}`,
            method: "GET"
        });
        self.displayAutocompleteSuggestions(
            response.suggestions.map(name => ({Name: name})),
            `🔍 ${response.suggestions.length} suggestion(s):`
        );
    }, 150));
    
    // CAS C: Clic sur suggestion
    $(document).on('click', '.autocomplete-suggestion-item', function(e) {
        const name = $(this).data('name');
        searchInput.val(name);
        self.handleSearch(name);  // Recherche auto!
    });
}
```



### 2. INTERFACE UTILISATEUR (15%)
**Responsabilité:** Design et expérience utilisateur

**Tâches réalisées:**
-  HTML structure (templates/index.html)
-  CSS custom (static/css/style.css)
  - Thème violet professionnel
  - Animations (fadeIn, slideIn, hover effects)
  - Responsive design
-  Intégration des cartes patients
- Modal pour détails complets
-  Boutons et dropdowns pour filtres ML

**Éléments créés:**
- Barre de recherche avec auto-complétion
- Cartes patients (design violet avec gradient)
- Zone de résultats avec animations
- Dropdowns pour filtres ML (3 dropdowns)
- Section visualisations ML
- Loading spinners


---

### 3. INTÉGRATION FRONTEND-BACKEND (10%)
**Responsabilité:** Communication AJAX et affichage

**Tâches réalisées:**
-  Fonction `handleVectorSearch()` (main.js lignes 550-600)
  - Récupération des 3 filtres ML
  - Envoi POST avec JSON
-  Fonction `displayVectorSearchResults()` (main.js lignes 602-750)
  - Affichage cartes patients
  - Affichage prédictions ML
  - Message avec statistiques
-  Fonction `loadClusterGraph()` (main.js lignes 1050-1150)
  - Chargement graphiques ML
  - Affichage images base64

**Code implémenté:**
```javascript
// main.js - lignes 550-600
async handleVectorSearch() {
    const query = $('#vectorSearchInput').val().trim();
    
    // RÉCUPÉRER LES 3 FILTRES ML
    const mlFilters = {
        risk_level: $('#mlFilterRisk').val() || null,
        wellness_score: $('#mlFilterScore').val() || null,
        cluster: $('#mlFilterCluster').val() || null
    };
    
    const response = await $.ajax({
        url: "/api/vector-search",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ 
            query: query,
            ml_filters: mlFilters  // Envoi des filtres!
        })
    });
    
    this.displayVectorSearchResults(response);
}
```



### 4. RECHERCHE CLASSIQUE (5%)
**Responsabilité:** Recherche textuelle traditionnelle

**Tâches réalisées:**
-  Route `/api/search` (app.py lignes 192-237)
- Regex MongoDB `.*query.*` (cherche partout dans le nom)
-  Fonction `handleSearch()` (main.js)
-  Affichage résultats avec détails complets

**Code implémenté:**
```python
# app.py - lignes 192-237
@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    query = request.args.get("query", "").strip()
    
    if not query:
        # Retourner tous (limité 100)
        results = collection.find().limit(100)
    else:
        # Recherche avec regex partout
        regex = {"$regex": f".*{re.escape(query)}.*", "$options": "i"}
        results = collection.find({"Name": regex}).limit(100)
    
    patients = [convert_objectid_to_str(p) for p in results]
    return jsonify({"patients": patients, "count": len(patients)})
```


---

### 5. TESTS ET DOCUMENTATION (5%)
**Responsabilité:** Qualité et documentation

**Tâches réalisées:**
-  Tests manuels de toutes les fonctionnalités
-  Vérification responsive (mobile, tablette, desktop)
- Test des cas limites (query vide, caractères spéciaux)
-  README.md (instructions d'installation)
- Commentaires dans le code JavaScript




