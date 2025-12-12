/* ===================================================================
 *  MindCare - JavaScript COMPLET 
 * =================================================================== */

(function ($) {
    'use strict';

    // ---------------------------------------------------------------
    // CONFIGURATION GLOBALE
    // ---------------------------------------------------------------
    const config = {
        apiBaseUrl: "",
        animationDuration: 800,
        mobileBreakpoint: 768
    };

    // ---------------------------------------------------------------
    // OUTILS
    // ---------------------------------------------------------------
    function debounce(func, wait) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    function throttle(func, limit) {
        let inThrottle;
        return (...args) => {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => (inThrottle = false), limit);
            }
        };
    }

    // ---------------------------------------------------------------
    // CLASSE PRINCIPALE
    // ---------------------------------------------------------------
    class MindCareApp {
        constructor() {
            this.patientsData = [];
            this.currentClusterModel = "kmeans";
            this.animationManager = new AnimationManager();
            this.autocompleteVisible = false;
            this.init();
        }

        // -----------------------------------------------------------
        // 🔵 INITIALISATION
        // -----------------------------------------------------------
        init() {
            this.bindEvents();
            this.initializeAnimations();
            this.setupClusterTabs();
            this.loadClusterGraph('kmeans');
            this.checkInitialStatus(); // B) Message de chargement
            this.setupAutocomplete(); // B) Auto-complétion
        }

        initializeAnimations() {
            this.typeEffect('.status-text', 'Connexion en cours...', 45);
        }

        // Effet d'écriture
        typeEffect(selector, text, speed) {
            const el = document.querySelector(selector);
            if (!el) return;
            let i = 0;
            el.innerText = "";
            const timer = setInterval(() => {
                el.innerText += text.charAt(i);
                i++;
                if (i >= text.length) clearInterval(timer);
            }, speed);
        }

        // -----------------------------------------------------------
        // B) VÉRIFIER LE STATUT INITIAL + MESSAGE
        // -----------------------------------------------------------
        async checkInitialStatus() {
            try {
                const response = await $.ajax({
                    url: "/api/status",
                    method: "GET"
                });

                if (response.status === "success") {
                    // Afficher le message dans le dashboard
                    const statusText = document.querySelector('.status-text');
                    if (statusText) {
                        statusText.innerText = response.message;
                        statusText.style.color = "#4CAF50";
                    }

                    // Mettre à jour le compteur de patients
                    const patientsCount = document.getElementById('patientsCount');
                    if (patientsCount && response.patient_count) {
                        this.animateCounter(patientsCount, 0, response.patient_count, 1500);
                    }

                    // Afficher une notification success
                    this.showNotification(response.message, "success");
                }
            } catch (err) {
                console.error("Erreur statut initial:", err);
                const statusText = document.querySelector('.status-text');
                if (statusText) {
                    statusText.innerText = " Connexion échouée";
                    statusText.style.color = "#f44336";
                }
            }
        }

        // Animation du compteur
        animateCounter(element, start, end, duration) {
            const range = end - start;
            const increment = range / (duration / 16);
            let current = start;

            const timer = setInterval(() => {
                current += increment;
                if (current >= end) {
                    element.innerText = Math.floor(end);
                    clearInterval(timer);
                } else {
                    element.innerText = Math.floor(current);
                }
            }, 16);
        }

        // -----------------------------------------------------------
        // B) AUTO-COMPLÉTION PENDANT LA FRAPPE (5 POINTS)
        // -----------------------------------------------------------
        setupAutocomplete() {
            const self = this;
            const searchInput = $('#searchInput');
            const searchResults = $('#searchResults');
            
            console.log(" setupAutocomplete() - Initialisé");

            // MESSAGE DE SUCCÈS AU CHARGEMENT
            searchResults.html(`
                <div class="success-message" style="padding: 1.5rem; text-align: center; color: #4CAF50; font-size: 1rem;">
                     Les données ont été chargées avec succès. Commencez à taper pour rechercher !
                </div>
            `);

            // CAS A) FOCUS SUR BARRE VIDE → 10 premiers patients
            searchInput.on('focus', async function() {
                const query = $(this).val().trim();
                console.log("🔵 Focus sur barre, query:", query);
                
                if (query.length === 0) {
                    console.log(" Chargement 10 premiers patients...");
                    try {
                        const response = await $.ajax({
                            url: `/api/search?query=`,
                            method: "GET"
                        });
                        
                        console.log(" Réponse reçue:", response);
                        if (response.results && response.results.length > 0) {
                            self.displayAutocompleteSuggestions(response.results.slice(0, 10), " 10 premiers patients:");
                        }
                    } catch (err) {
                        console.error(" Erreur chargement 10 premiers:", err);
                    }
                }
            });

            // CAS B) AUTO-COMPLÉTION PENDANT LA FRAPPE (EN TEMPS RÉEL!)
            searchInput.on('input', debounce(async function() {
                const query = $(this).val().trim();
                console.log(" Input détecté, query:", query);

                // Si vide, afficher les 10 premiers
                if (query.length === 0) {
                    console.log(" Query vide, trigger focus");
                    searchInput.trigger('focus');
                    return;
                }

                // PENDANT LA FRAPPE → SUGGESTIONS EN TEMPS RÉEL
                console.log(" Appel API autocomplete pour:", query);
                try {
                    const response = await $.ajax({
                        url: `/api/autocomplete?query=${encodeURIComponent(query)}`,
                        method: "GET"
                    });

                    console.log(" Suggestions reçues:", response.suggestions);
                    
                    if (response.suggestions && response.suggestions.length > 0) {
                        console.log(` Affichage de ${response.suggestions.length} suggestions`);
                        self.displayAutocompleteSuggestions(
                            response.suggestions.map(name => ({Name: name})), 
                            ` ${response.suggestions.length} suggestion(s) pour "${query}":`
                        );
                    } else {
                        console.log(" Aucune suggestion trouvée");
                        searchResults.html('<div style="padding: 1rem; color: #999; text-align: center;">Aucune suggestion trouvée</div>');
                    }
                } catch (err) {
                    console.error(" Erreur autocomplete:", err);
                }
            }, 150)); // 150ms pour réactivité

            // CLIC sur suggestion → RECHERCHE COMPLÈTE AUTOMATIQUE
            $(document).on('click', '.autocomplete-suggestion-item', function(e) {
                e.preventDefault();
                const name = $(this).data('name');
                console.log(" Clic sur suggestion:", name);
                
                searchInput.val(name);
                console.log(" Lancement recherche pour:", name);
                
                // LANCER RECHERCHE AUTOMATIQUEMENT (sans Enter!)
                self.handleSearch(name);
            });
            
            console.log(" setupAutocomplete() - Tous les events bindés");
        }

        // Afficher suggestions d'auto-complétion
        displayAutocompleteSuggestions(items, title) {
            console.log(" displayAutocompleteSuggestions:", items.length, "items");
            
            let html = `
                <div style="background: rgba(124, 58, 237, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 2px solid #7c3aed;">
                    <h4 style="color: #7c3aed; margin: 0; font-size: 1rem;">${title}</h4>
                    <p style="color: #999; font-size: 0.85rem; margin: 0.5rem 0 0 0;"> Cliquez sur un patient pour voir ses détails (pas besoin d'Enter!)</p>
                </div>
            `;
            
            items.forEach(item => {
                const name = item.Name || item;
                html += `
                    <div class="autocomplete-suggestion-item" 
                         data-name="${name}"
                         style="
                            padding: 1.2rem;
                            margin-bottom: 0.5rem;
                            cursor: pointer;
                            border: 2px solid #7c3aed;
                            border-radius: 8px;
                            transition: all 0.2s;
                            background: rgba(124, 58, 237, 0.1);
                            display: flex;
                            align-items: center;
                            gap: 0.8rem;
                        " 
                        onmouseover="this.style.background='rgba(124, 58, 237, 0.3)'; this.style.transform='translateX(5px)'; this.style.boxShadow='0 4px 12px rgba(124, 58, 237, 0.4)'"
                        onmouseout="this.style.background='rgba(124, 58, 237, 0.1)'; this.style.transform='translateX(0)'; this.style.boxShadow='none'"
                    >
                        <span style="color: #7c3aed; font-size: 1.5rem;">👤</span>
                        <span style="color: white; font-weight: 600; font-size: 1.05rem;">${name}</span>
                        <span style="margin-left: auto; color: #7c3aed; font-size: 0.9rem; font-weight: 500;">Cliquez ici →</span>
                    </div>
                `;
            });
            
            $('#searchResults').html(html);
            console.log("✅ Suggestions affichées dans #searchResults");
        }

        // -----------------------------------------------------------
        // GESTION DES ÉVÉNEMENTS
        // -----------------------------------------------------------
        bindEvents() {
            // Recherche SEULEMENT sur Enter (pas à chaque lettre)
            $('#searchInput').on('keypress', (e) => {
                if (e.which === 13) { // Enter key
                    this.handleSearch(e);
                }
            });
            
            $('#vectorSearchBtn').on('click', this.handleVectorSearch.bind(this));
            $('#calculateEDF').on('click', this.handleCalculateEDF.bind(this));

            // CRUD dans les résultats
            $(document).on('click', '.view-patient', this.handleViewPatient.bind(this));
            $(document).on('click', '.edit-patient', this.handleEditPatient.bind(this));
            $(document).on('click', '.delete-patient', this.handleDeletePatient.bind(this)); // D) Nouveau
            
            // Boutons de visualisation des clusters
            $(document).on('click', '.cluster-btn', this.handleClusterClick.bind(this));
            
            // PAGINATION - Nouveau!
            $(document).on('click', '.pagination-btn', (e) => {
                const page = $(e.currentTarget).data('page');
                const type = $(e.currentTarget).data('type');
                if (page && type) {
                    this.handlePagination(page, type);
                }
            });
            
            // CRUD page de gestion
            $('#btnNewPatient').on('click', this.handleCreatePatient.bind(this));
            $('#btnExportPatients').on('click', () => this.exportPatients("json"));
            $('#btnOpenPatients').on('click', () =>
                $('html,body').animate({ scrollTop: $('#searchResults').offset().top - 60 }, 600)
            );
        }

        // -----------------------------------------------------------
        // C) RECHERCHE CLASSIQUE (AFFICHE 6-7 CHAMPS)
        // -----------------------------------------------------------
        async handleSearch(eventOrQuery) {
            // Accepter soit un event, soit une query directe
            let query;
            if (typeof eventOrQuery === 'string') {
                query = eventOrQuery.trim();
            } else {
                query = $(eventOrQuery.target).val().trim();
            }

            if (query.length === 0) {
                $('#searchResults').html('<p class="placeholder">Tapez pour rechercher...</p>');
                return;
            }

            if (query.length === 1) return;

            try {
                // AFFICHER MESSAGE DE CHARGEMENT PROGRESSIF
                const searchResults = $('#searchResults');
                searchResults.html(`
                    <div style="padding: 2rem; text-align: center; color: #7c3aed;">
                        <div class="spinner" style="margin: 0 auto 1rem;"></div>
                        <p> Recherche en cours...</p>
                        <p style="color: #999; font-size: 0.9rem;">Les premiers résultats apparaîtront bientôt</p>
                    </div>
                `);
                
                // Reset pagination pour nouvelle recherche
                this.currentSearchPage = 1;

                // LANCER LA RECHERCHE
                const response = await $.ajax({
                    url: "/api/search",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify({ query })
                });

                this.patientsData = response.patients || [];
                
                // AFFICHAGE PROGRESSIF DES RÉSULTATS
                this.displaySearchResultsProgressively(response);

            } catch (err) {
                this.showError("Échec de recherche : " + (err.responseJSON?.error || err.message), "#searchResults");
            }
        }

        // NOUVELLE FONCTION: Affichage progressif des résultats
        displaySearchResultsProgressively(response) {
            const container = $('#searchResults');
            
            if (!response.patients || response.patients.length === 0) {
                container.html(`<p class="no-results">Aucun résultat trouvé.</p>`);
                return;
            }

            const RESULTS_PER_PAGE = 15;
            const BATCH_SIZE = 5;  // Afficher 5 résultats à la fois
            const totalResults = response.patients.length;
            const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE);
            
            // Initialiser la page courante
            if (!this.currentSearchPage) this.currentSearchPage = 1;
            
            // Calculer les résultats de la page courante
            const startIndex = (this.currentSearchPage - 1) * RESULTS_PER_PAGE;
            const endIndex = Math.min(startIndex + RESULTS_PER_PAGE, totalResults);
            const pageResults = response.patients.slice(startIndex, endIndex);
            
            // HEADER avec compte total
            let html = `
                <div class="results-header" style="padding: 1.5rem; background: rgba(124, 58, 237, 0.1); border-radius: 8px; margin-bottom: 1.5rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #7c3aed; font-size: 1.2rem;">${response.message || 'Résultats de recherche'}</h4>
                    <div style="color: #999; font-size: 0.95rem;">
                        <b style="color: white;">${totalResults}</b> résultat(s) trouvé(s) • 
                        Page <b style="color: white;">${this.currentSearchPage}</b> sur <b style="color: white;">${totalPages}</b>
                    </div>
                </div>
                <div id="progressiveResultsContainer"></div>
            `;
            
            container.html(html);
            
            // AFFICHER LES RÉSULTATS PROGRESSIVEMENT (5 par 5)
            const progressiveContainer = $('#progressiveResultsContainer');
            let displayedCount = 0;
            
            const displayNextBatch = () => {
                const batchEnd = Math.min(displayedCount + BATCH_SIZE, pageResults.length);
                const batch = pageResults.slice(displayedCount, batchEnd);
                
                batch.forEach(patient => {
                    const card = this.createPatientCard(patient);
                    progressiveContainer.append(card);
                    
                    // Animation d'apparition
                    $(card).hide().fadeIn(300);
                });
                
                displayedCount = batchEnd;
                
                // Continuer si il reste des résultats
                if (displayedCount < pageResults.length) {
                    setTimeout(displayNextBatch, 100);  // 100ms entre chaque batch
                } else {
                    // AJOUTER PAGINATION À LA FIN
                    if (totalPages > 1) {
                        progressiveContainer.append(this.createPagination(this.currentSearchPage, totalPages, 'search'));
                    }
                }
            };
            
            // DÉMARRER L'AFFICHAGE PROGRESSIF
            displayNextBatch();
            
            // Sauvegarder pour pagination
            this.allSearchResults = response.patients;
        }

        displaySearchResults(response) {
            // Utiliser l'affichage progressif pour tout
            this.displaySearchResultsProgressively(response);
        }

        // Créer les boutons de pagination
        createPagination(currentPage, totalPages, type) {
            let html = `
                <div class="pagination" style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; margin-top: 2rem; padding: 1rem;">
                    <button class="btn pagination-btn" data-page="1" data-type="${type}" ${currentPage === 1 ? 'disabled' : ''}>
                        « Première
                    </button>
                    <button class="btn pagination-btn" data-page="${currentPage - 1}" data-type="${type}" ${currentPage === 1 ? 'disabled' : ''}>
                        ‹ Précédent
                    </button>
                    <div style="color: white; margin: 0 1rem;">
                        Page <b>${currentPage}</b> sur <b>${totalPages}</b>
                    </div>
                    <button class="btn pagination-btn" data-page="${currentPage + 1}" data-type="${type}" ${currentPage === totalPages ? 'disabled' : ''}>
                        Suivant ›
                    </button>
                    <button class="btn pagination-btn" data-page="${totalPages}" data-type="${type}" ${currentPage === totalPages ? 'disabled' : ''}>
                        Dernière »
                    </button>
                </div>
            `;
            return html;
        }

        // Gérer les clics de pagination
        handlePagination(page, type) {
            if (type === 'search') {
                this.currentSearchPage = parseInt(page);
                this.displaySearchResults({ patients: this.allSearchResults });
                // Scroll vers le haut
                $('#searchResults').get(0).scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else if (type === 'vector') {
                this.currentVectorPage = parseInt(page);
                this.displayVectorSearchResults({ results: this.allVectorResults });
                // Scroll vers le haut
                $('#vectorSearchResults').get(0).scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        // C) CORRECTION: Afficher 10 CHAMPS (tous les principaux)
        createPatientCard(p) {
            // Formater le revenu avec séparateur de milliers
            const income = p.Income ? new Intl.NumberFormat('fr-CA', { 
                style: 'currency', 
                currency: 'CAD',
                maximumFractionDigits: 0
            }).format(p.Income) : "—";

            return `
                <div class="patient-card">
                    <div class="header">
                        <h4>${p.Name || "Nom inconnu"}</h4>
                        <span class="age-badge">${p.Age || "?"} ans</span>
                    </div>

                    <div class="details">
                        <p><b>Statut marital:</b> ${p["Marital Status"] || "—"}</p>
                        <p><b>Emploi:</b> ${p["Employment Status"] || "—"}</p>
                        <p><b>Éducation:</b> ${p["Education Level"] || "—"}</p>
                        <p><b>Enfants:</b> ${p["Number of Children"] || 0}</p>
                        <p><b>Revenu:</b> ${income}</p>
                        <p><b>Tabagisme:</b> ${p["Smoking Status"] || "—"}</p>
                        <p><b>Sommeil:</b> ${p["Sleep Patterns"] || "—"}</p>
                        <p><b>Activité physique:</b> ${p["Physical Activity Level"] || "—"}</p>
                        <p><b>Alcool:</b> ${p["Alcohol Consumption"] || "—"}</p>
                        <p><b>Alimentation:</b> ${p["Dietary Habits"] || "—"}</p>
                    </div>

                    <div class="actions">
                        <button class="button small view-patient" data-id="${p._id}">
                            <i class="icon-eye"></i> Voir
                        </button>
                        <button class="button small stroke edit-patient" data-id="${p._id}">
                            <i class="icon-edit"></i> Modifier
                        </button>
                        <button class="button small danger delete-patient" data-id="${p._id}">
                            <i class="icon-trash"></i> Supprimer
                        </button>
                    </div>
                </div>
            `;
        }

        // -----------------------------------------------------------
        // A) RECHERCHE VECTORIELLE (CORRIGÉE)
        // -----------------------------------------------------------
        async handleVectorSearch() {
            const query = $('#vectorSearchInput').val().trim();
            if (!query) {
                this.showError("Veuillez entrer une description.", "#vectorSearchResults");
                return;
            }

            try {
                this.showLoading("#vectorSearchResults");
                
                // Reset pagination pour nouvelle recherche
                this.currentVectorPage = 1;
                
                // RÉCUPÉRER LES FILTRES ML (6 POINTS - CATÉGORISATION)
                const mlFilters = {
                    risk_level: $('#mlFilterRisk').val() || null,      // Random Forest
                    wellness_score: $('#mlFilterScore').val() || null,  // XGBoost
                    cluster: $('#mlFilterCluster').val() || null        // K-Means
                };

                const response = await $.ajax({
                    url: "/api/vector-search",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify({ 
                        query: query,
                        ml_filters: mlFilters  // Envoyer filtres au backend
                    })
                });

                this.displayVectorSearchResults(response);

            } catch (err) {
                console.error("Erreur recherche vectorielle:", err);
                this.showError("Erreur recherche : " + (err.responseJSON?.error || err.message), "#vectorSearchResults");
            } finally {
                this.hideLoading("#vectorSearchResults");
            }
        }

        displayVectorSearchResults(response) {
            const container = $('#vectorSearchResults');
            container.html("");

            if (!response.results?.length) {
                container.html("<p>Aucun résultat trouvé.</p>");
                return;
            }

            // Pagination: 15 résultats par page
            const RESULTS_PER_PAGE = 15;
            const totalResults = response.results.length;
            const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE);
            
            // Page actuelle
            if (!this.currentVectorPage) this.currentVectorPage = 1;
            
            const startIndex = (this.currentVectorPage - 1) * RESULTS_PER_PAGE;
            const endIndex = Math.min(startIndex + RESULTS_PER_PAGE, totalResults);
            const pageResults = response.results.slice(startIndex, endIndex);

            // En-tête avec nombre de résultats et pagination
            let html = `
                <div class="results-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <h4 style="color: white; margin: 0;">${response.message || 'Résultats de recherche vectorielle'}</h4>
                    <div style="color: white;">
                        <b>${totalResults}</b> résultat${totalResults > 1 ? 's' : ''} trouvé${totalResults > 1 ? 's' : ''}
                    </div>
                </div>
            `;

            // Afficher les résultats de la page actuelle
            pageResults.forEach(r => {
                html += this.createVectorResultCard(r);
            });

            // Pagination en bas (si plus d'une page)
            if (totalPages > 1) {
                html += this.createPagination(this.currentVectorPage, totalPages, 'vector');
            }

            container.html(html);
            
            // Sauvegarder les données pour pagination
            this.allVectorResults = response.results;
        }

        createVectorResultCard(r) {
            const info = r.patient_info || {};
            const ml = r.ml_predictions || {};
            const raw = r.raw || {};

            return `
                <div class="vector-result-card">
                    <div class="result-header">
                        <h4>${info.name || raw.Name || "Inconnu"}</h4>
                        <span class="similarity-badge">${(info.similarity_score * 100).toFixed(1)}% similaire</span>
                    </div>
                    <div class="result-details">
                        <p><b>Âge:</b> ${info.age || raw.Age || "?"} ans</p>
                        <p><b>Genre:</b> ${info.gender || raw.Gender || "?"}</p>
                        <p><b>Profession:</b> ${info.profession || raw.Profession || "?"}</p>
                        <p><b>Niveau de risque:</b> <span class="risk-badge ${ml.risk_category === 'Risque Élevé' ? 'risk-high' : 'risk-low'}">${ml.risk_category || "?"}</span></p>
                        <p><b>Score bien-être:</b> ${ml.wellness_score !== undefined && ml.wellness_score !== "N/A" ? ml.wellness_score + "/100" : "?"}</p>
                        <p><b>Cluster:</b> ${ml.cluster_label || ml.cluster_group || "?"}</p>
                        <p><b>Probabilité dépression:</b> ${ml.risk_probability !== undefined ? (ml.risk_probability * 100).toFixed(1) + "%" : "?"}</p>
                    </div>
                    <div class="actions">
                        <button class="button small view-patient" data-id="${raw._id || info.id}">Voir détails complets</button>
                    </div>
                </div>
            `;
        }

        // -----------------------------------------------------------
        // D) VOIR PATIENT (TOUS LES 16 CHAMPS - ORGANISÉS PAR SECTIONS)
        // -----------------------------------------------------------
        async handleViewPatient(e) {
            const id = $(e.currentTarget).data('id');
            if (!id) return;

            try {
                const response = await $.ajax({
                    url: `/api/patient/${id}`,
                    method: "GET"
                });

                const p = response.patient;

                // Formater le revenu
                const income = p.Income ? new Intl.NumberFormat('fr-CA', { 
                    style: 'currency', 
                    currency: 'CAD',
                    maximumFractionDigits: 2
                }).format(p.Income) : "Non spécifié";

                // D) CORRECTION: Afficher TOUS les champs organisés par sections
                Swal.fire({
                    title: `<i class="icon-user"></i> ${p.Name || "Patient"}`,
                    html: `
                        <div style="text-align: left; max-height: 500px; overflow-y: auto; padding: 10px;">
                            
                            <!-- Section Informations Générales -->
                            <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="color: #667eea; margin: 0 0 10px 0; font-size: 1rem;">👤 Informations Générales</h4>
                                <p style="margin: 5px 0;"><b>Âge:</b> ${p.Age || "—"} ans</p>
                                <p style="margin: 5px 0;"><b>Statut marital:</b> ${p["Marital Status"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Nombre d'enfants:</b> ${p["Number of Children"] || 0}</p>
                            </div>

                            <!-- Section Situation Professionnelle -->
                            <div style="background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="color: #f5576c; margin: 0 0 10px 0; font-size: 1rem;">💼 Situation Professionnelle</h4>
                                <p style="margin: 5px 0;"><b>Emploi:</b> ${p["Employment Status"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Éducation:</b> ${p["Education Level"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Revenu annuel:</b> ${income}</p>
                            </div>

                            <!-- Section Style de Vie -->
                            <div style="background: linear-gradient(135deg, #4facfe15 0%, #00f2fe15 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="color: #4facfe; margin: 0 0 10px 0; font-size: 1rem;">🏃 Style de Vie</h4>
                                <p style="margin: 5px 0;"><b>Tabagisme:</b> ${p["Smoking Status"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Consommation d'alcool:</b> ${p["Alcohol Consumption"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Activité physique:</b> ${p["Physical Activity Level"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Habitudes alimentaires:</b> ${p["Dietary Habits"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Qualité du sommeil:</b> ${p["Sleep Patterns"] || "—"}</p>
                            </div>

                            <!-- Section Historique Médical -->
                            <div style="background: linear-gradient(135deg, #fa709a15 0%, #fee14015 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="color: #fa709a; margin: 0 0 10px 0; font-size: 1rem;">🏥 Historique Médical</h4>
                                <p style="margin: 5px 0;"><b>Historique maladie mentale:</b> ${p["History of Mental Illness"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Historique abus de substances:</b> ${p["History of Substance Abuse"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Historique familial de dépression:</b> ${p["Family History of Depression"] || "—"}</p>
                                <p style="margin: 5px 0;"><b>Conditions médicales chroniques:</b> ${p["Chronic Medical Conditions"] || "—"}</p>
                            </div>

                            <!-- ID MongoDB -->
                            <div style="text-align: center; margin-top: 15px; padding-top: 10px; border-top: 1px solid #e0e0e0;">
                                <small style="color: #999; font-family: monospace;">ID: ${p._id}</small>
                            </div>

                        </div>
                    `,
                    width: 700,
                    confirmButtonText: "Fermer",
                    confirmButtonColor: "#667eea"
                });

            } catch (err) {
                Swal.fire("Erreur", err.responseJSON?.error || "Patient non trouvé", "error");
            }
        }

        // -----------------------------------------------------------
        // D) MODIFIER PATIENT (PLUS DE CHAMPS MODIFIABLES)
        // -----------------------------------------------------------
        async handleEditPatient(e) {
            const id = $(e.currentTarget).data('id');

            try {
                const response = await $.ajax({
                    url: `/api/patient/${id}`,
                    method: "GET"
                });

                const p = response.patient;

                const { value } = await Swal.fire({
                    title: ` Modifier: ${p.Name}`,
                    html: `
                        <div style="max-height: 500px; overflow-y: auto; text-align: left; padding: 10px;">
                            
                            <h4 style="color: #667eea; font-size: 0.95rem; margin: 15px 0 8px 0;"> Informations de base</h4>
                            <label style="font-size: 0.85rem; color: #666;">Âge:</label>
                            <input id="age" class="swal2-input" type="number" value="${p.Age || ''}" style="width: 90%;">
                            
                            <label style="font-size: 0.85rem; color: #666;">Statut marital: <span style="color: #999;">(Single, Married, Divorced, Widowed)</span></label>
                            <input id="marital" class="swal2-input" type="text" value="${p["Marital Status"] || ''}" style="width: 90%;" placeholder="ex: Married">
                            
                            <label style="font-size: 0.85rem; color: #666;">Nombre d'enfants:</label>
                            <input id="children" class="swal2-input" type="number" value="${p["Number of Children"] || 0}" style="width: 90%;">
                            
                            <h4 style="color: #667eea; font-size: 0.95rem; margin: 15px 0 8px 0;">💼 Situation professionnelle</h4>
                            <label style="font-size: 0.85rem; color: #666;">Emploi: <span style="color: #999;">(Employed, Unemployed, Student, Retired)</span></label>
                            <input id="employment" class="swal2-input" type="text" value="${p["Employment Status"] || ''}" style="width: 90%;" placeholder="ex: Employed">
                            
                            <label style="font-size: 0.85rem; color: #666;">Éducation: <span style="color: #999;">(High School, Bachelor's Degree, Master's Degree, PhD)</span></label>
                            <input id="education" class="swal2-input" type="text" value="${p["Education Level"] || ''}" style="width: 90%;" placeholder="ex: Bachelor's Degree">
                            
                            <label style="font-size: 0.85rem; color: #666;">Revenu annuel ($):</label>
                            <input id="income" class="swal2-input" type="number" value="${p.Income || 0}" style="width: 90%;">
                            
                            <h4 style="color: #667eea; font-size: 0.95rem; margin: 15px 0 8px 0;">🏃 Style de vie</h4>
                            <label style="font-size: 0.85rem; color: #666;">Tabagisme: <span style="color: #999;">(Non-smoker, Former, Current)</span></label>
                            <input id="smoking" class="swal2-input" type="text" value="${p["Smoking Status"] || ''}" style="width: 90%;" placeholder="ex: Non-smoker">
                            
                            <label style="font-size: 0.85rem; color: #666;">Consommation d'alcool: <span style="color: #999;">(Low, Moderate, High)</span></label>
                            <input id="alcohol" class="swal2-input" type="text" value="${p["Alcohol Consumption"] || ''}" style="width: 90%;" placeholder="ex: Low">
                            
                            <label style="font-size: 0.85rem; color: #666;">Activité physique: <span style="color: #999;">(Sedentary, Moderate, Active)</span></label>
                            <input id="activity" class="swal2-input" type="text" value="${p["Physical Activity Level"] || ''}" style="width: 90%;" placeholder="ex: Moderate">
                            
                            <label style="font-size: 0.85rem; color: #666;">Qualité du sommeil: <span style="color: #999;">(Poor, Fair, Good, Excellent)</span></label>
                            <input id="sleep" class="swal2-input" type="text" value="${p["Sleep Patterns"] || ''}" style="width: 90%;" placeholder="ex: Good">
                            
                            <label style="font-size: 0.85rem; color: #666;">Alimentation: <span style="color: #999;">(Unhealthy, Moderate, Healthy)</span></label>
                            <input id="dietary" class="swal2-input" type="text" value="${p["Dietary Habits"] || ''}" style="width: 90%;" placeholder="ex: Healthy">
                            
                        </div>
                    `,
                    width: 650,
                    showCancelButton: true,
                    confirmButtonText: " Enregistrer",
                    cancelButtonText: "Annuler",
                    confirmButtonColor: "#667eea",
                    cancelButtonColor: "#95a5a6",
                    preConfirm: () => {
                        return {
                            Age: Number($('#age').val()),
                            "Marital Status": $('#marital').val(),
                            "Number of Children": Number($('#children').val()),
                            "Employment Status": $('#employment').val(),
                            "Education Level": $('#education').val(),
                            "Income": Number($('#income').val()),
                            "Smoking Status": $('#smoking').val(),
                            "Alcohol Consumption": $('#alcohol').val(),
                            "Physical Activity Level": $('#activity').val(),
                            "Sleep Patterns": $('#sleep').val(),
                            "Dietary Habits": $('#dietary').val()
                        };
                    }
                });

                if (value) {
                    await $.ajax({
                        url: `/api/patient/${id}`,
                        method: "PUT",
                        contentType: "application/json",
                        data: JSON.stringify(value)
                    });
                    
                    Swal.fire({
                        title: "Succès!",
                        text: " Patient mis à jour avec succès",
                        icon: "success",
                        confirmButtonColor: "#667eea"
                    });
                    
                    // Rafraîchir la recherche
                    this.handleSearch({ target: $('#searchInput')[0] });
                }
            } catch (err) {
                Swal.fire("Erreur", err.responseJSON?.error || "Erreur de modification", "error");
            }
        }

        // -----------------------------------------------------------
        // D) SUPPRIMER PATIENT (NOUVEAU)
        // -----------------------------------------------------------
        async handleDeletePatient(e) {
            const id = $(e.currentTarget).data('id');
            if (!id) return;

            const result = await Swal.fire({
                title: "Êtes-vous sûr?",
                text: "Cette action est irréversible!",
                icon: "warning",
                showCancelButton: true,
                confirmButtonColor: "#d33",
                cancelButtonColor: "#3085d6",
                confirmButtonText: "Oui, supprimer",
                cancelButtonText: "Annuler"
            });

            if (result.isConfirmed) {
                try {
                    await $.ajax({
                        url: `/api/patient/${id}`,
                        method: "DELETE"
                    });

                    Swal.fire("Supprimé!", " Le patient a été supprimé.", "success");
                    
                    // Rafraîchir la recherche
                    this.handleSearch({ target: $('#searchInput')[0] });

                } catch (err) {
                    Swal.fire("Erreur", err.responseJSON?.error || "Erreur de suppression", "error");
                }
            }
        }

        // -----------------------------------------------------------
        // D) CRÉER PATIENT (TOUS LES 16 CHAMPS OBLIGATOIRES)
        // -----------------------------------------------------------
        async handleCreatePatient() {
            const { value } = await Swal.fire({
                title: " Nouveau patient - Tous les champs obligatoires",
                html: `
                    <div style="text-align: left; max-height: 500px; overflow-y: auto; padding: 10px;">
                        
                        <h4 style="color: #667eea; font-size: 1rem; margin: 15px 0 8px 0;"> Informations de base</h4>
                        
                        <label style="font-size: 0.85rem; color: #666;">Nom complet *:</label>
                        <input id="name" class="swal2-input" placeholder="Ex: John Doe" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Âge *:</label>
                        <input id="age" class="swal2-input" type="number" placeholder="Ex: 35" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Statut marital *: <span style="color: #999; font-size: 0.75rem;">(Single, Married, Divorced, Widowed)</span></label>
                        <input id="marital" class="swal2-input" placeholder="Ex: Married" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Nombre d'enfants *:</label>
                        <input id="children" class="swal2-input" type="number" placeholder="Ex: 0" value="0" style="width: 90%;">
                        
                        <h4 style="color: #667eea; font-size: 1rem; margin: 15px 0 8px 0;">💼 Situation professionnelle</h4>
                        
                        <label style="font-size: 0.85rem; color: #666;">Emploi *: <span style="color: #999; font-size: 0.75rem;">(Employed, Unemployed, Student, Retired)</span></label>
                        <input id="employment" class="swal2-input" placeholder="Ex: Employed" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Éducation *: <span style="color: #999; font-size: 0.75rem;">(High School, Bachelor's Degree, Master's Degree, PhD)</span></label>
                        <input id="education" class="swal2-input" placeholder="Ex: Bachelor's Degree" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Revenu annuel ($) *:</label>
                        <input id="income" class="swal2-input" type="number" placeholder="Ex: 50000" style="width: 90%;">
                        
                        <h4 style="color: #667eea; font-size: 1rem; margin: 15px 0 8px 0;">🏃 Style de vie</h4>
                        
                        <label style="font-size: 0.85rem; color: #666;">Tabagisme *: <span style="color: #999; font-size: 0.75rem;">(Non-smoker, Former, Current)</span></label>
                        <input id="smoking" class="swal2-input" placeholder="Ex: Non-smoker" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Consommation d'alcool *: <span style="color: #999; font-size: 0.75rem;">(Low, Moderate, High)</span></label>
                        <input id="alcohol" class="swal2-input" placeholder="Ex: Low" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Activité physique *: <span style="color: #999; font-size: 0.75rem;">(Sedentary, Moderate, Active)</span></label>
                        <input id="activity" class="swal2-input" placeholder="Ex: Moderate" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Habitudes alimentaires *: <span style="color: #999; font-size: 0.75rem;">(Unhealthy, Moderate, Healthy)</span></label>
                        <input id="dietary" class="swal2-input" placeholder="Ex: Healthy" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Qualité du sommeil *: <span style="color: #999; font-size: 0.75rem;">(Poor, Fair, Good, Excellent)</span></label>
                        <input id="sleep" class="swal2-input" placeholder="Ex: Good" style="width: 90%;">
                        
                        <h4 style="color: #667eea; font-size: 1rem; margin: 15px 0 8px 0;">🏥 Historique médical</h4>
                        
                        <label style="font-size: 0.85rem; color: #666;">Historique maladie mentale *: <span style="color: #999; font-size: 0.75rem;">(Yes, No)</span></label>
                        <input id="mental" class="swal2-input" placeholder="Ex: No" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Historique abus de substances *: <span style="color: #999; font-size: 0.75rem;">(Yes, No)</span></label>
                        <input id="substance" class="swal2-input" placeholder="Ex: No" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Historique familial de dépression *: <span style="color: #999; font-size: 0.75rem;">(Yes, No)</span></label>
                        <input id="family" class="swal2-input" placeholder="Ex: No" style="width: 90%;">
                        
                        <label style="font-size: 0.85rem; color: #666;">Conditions médicales chroniques *: <span style="color: #999; font-size: 0.75rem;">(Yes, No)</span></label>
                        <input id="chronic" class="swal2-input" placeholder="Ex: No" style="width: 90%;">
                        
                    </div>
                `,
                width: 700,
                showCancelButton: true,
                confirmButtonText: " Créer le patient",
                cancelButtonText: "Annuler",
                confirmButtonColor: "#667eea",
                cancelButtonColor: "#95a5a6",
                preConfirm: () => {
                    const name = $('#name').val().trim();
                    const age = $('#age').val();
                    const marital = $('#marital').val();
                    const employment = $('#employment').val();
                    const education = $('#education').val();
                    const income = $('#income').val();
                    const smoking = $('#smoking').val();
                    const alcohol = $('#alcohol').val();
                    const activity = $('#activity').val();
                    const dietary = $('#dietary').val();
                    const sleep = $('#sleep').val();
                    const mental = $('#mental').val();
                    const substance = $('#substance').val();
                    const family = $('#family').val();
                    const chronic = $('#chronic').val();
                    
                    // Validation: TOUS les champs sont obligatoires
                    if (!name || !age || !marital || !employment || !education || !income || 
                        !smoking || !alcohol || !activity || !dietary || !sleep || 
                        !mental || !substance || !family || !chronic) {
                        Swal.showValidationMessage(' Tous les champs marqués * sont obligatoires');
                        return false;
                    }

                    return {
                        Name: name,
                        Age: Number(age),
                        "Marital Status": marital,
                        "Employment Status": employment,
                        "Education Level": education,
                        "Number of Children": Number($('#children').val()) || 0,
                        Income: Number(income),
                        "Smoking Status": smoking,
                        "Alcohol Consumption": alcohol,
                        "Physical Activity Level": activity,
                        "Dietary Habits": dietary,
                        "Sleep Patterns": sleep,
                        "History of Mental Illness": mental,
                        "History of Substance Abuse": substance,
                        "Family History of Depression": family,
                        "Chronic Medical Conditions": chronic
                    };
                }
            });

            if (value) {
                try {
                    const response = await $.ajax({
                        url: "/api/patient",
                        method: "POST",
                        contentType: "application/json",
                        data: JSON.stringify(value)
                    });

                    Swal.fire("Succès", response.message || " Patient créé avec succès", "success");
                    
                    // Optionnel: Rafraîchir la recherche pour voir le nouveau patient
                    this.handleSearch({ target: $('#searchInput')[0] });

                } catch (err) {
                    Swal.fire("Erreur", err.responseJSON?.error || "Erreur de création", "error");
                }
            }
        }

        // -----------------------------------------------------------
        // EDF (AVEC INTERPRÉTATION)
        // -----------------------------------------------------------
        async handleCalculateEDF() {
            const variable = $('#edfVariable').val();

            try {
                this.showLoading("#edfResults");

                const response = await $.ajax({
                    url: "/api/statistics/edf",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify({ variable })
                });

                // Backend retourne DÉJÀ le format data:image/png;base64,... (PAS de double préfixe!)
                const imgSrc = response.edf_image;
                const stats = response.statistics;

                let html = `
                    <div class="edf-result">
                        <div style="margin-bottom: 2rem;">
                            <img src="${imgSrc}" alt="EDF Graph" style="width: 100%; max-width: 100%; height: auto; border-radius: 8px; display: block;">
                        </div>
                        
                        <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                            <div class="stat-item" style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;">
                                <span class="stat-label" style="color: #999; font-size: 0.85rem;">Moyenne</span>
                                <span class="stat-value" style="color: white; font-size: 1.5rem; font-weight: bold; display: block;">${stats.mean}</span>
                            </div>
                            <div class="stat-item" style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;">
                                <span class="stat-label" style="color: #999; font-size: 0.85rem;">Médiane</span>
                                <span class="stat-value" style="color: white; font-size: 1.5rem; font-weight: bold; display: block;">${stats.median}</span>
                            </div>
                            <div class="stat-item" style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;">
                                <span class="stat-label" style="color: #999; font-size: 0.85rem;">Q1 / Q3</span>
                                <span class="stat-value" style="color: white; font-size: 1.5rem; font-weight: bold; display: block;">${stats.q1} / ${stats.q3}</span>
                            </div>
                        </div>

                        ${response.interpretation ? `
                            <div class="interpretation-box" style="background: white; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #7c3aed; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <h4 style="color: #7c3aed; margin-top: 0;">📊 Interprétation</h4>
                                <div class="interpretation-text" style="color: #1a1a1a; line-height: 1.8; font-size: 0.95rem;">${response.interpretation.replace(/\n/g, '<br>')}</div>
                            </div>
                        ` : ''}
                    </div>
                `;

                $('#edfResults').html(html);

            } catch (err) {
                this.showError("Erreur EDF : " + (err.responseJSON?.error || err.message), "#edfResults");
            } finally {
                this.hideLoading("#edfResults");
            }
        }

        // -----------------------------------------------------------
        // EXPORT
        // -----------------------------------------------------------
        exportPatients(format) {
            if (!this.patientsData?.length) {
                this.showError("Aucune donnée à exporter.");
                return;
            }

            const blob = new Blob(
                [JSON.stringify(this.patientsData, null, 2)],
                { type: "application/json" }
            );

            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `patients_${new Date().toISOString().split('T')[0]}.json`;
            link.click();

            this.showNotification(" Export réussi!", "success");
        }

        // -----------------------------------------------------------
        // UTILITAIRES
        // -----------------------------------------------------------
        showLoading(sel) {
            $(sel).html('<div class="loading-spinner"><div class="spinner"></div><p>Chargement...</p></div>');
        }

        hideLoading(sel) {
            // Ne rien faire - le contenu remplacera le spinner
        }

        showError(msg, sel = "body") {
            const $container = $(sel);
            $container.html(`
                <div class="error-message">
                    <i class="icon-warning"></i>
                    <span>${msg}</span>
                </div>
            `);
        }

        showNotification(msg, type = "info") {
            const bgColor = type === "success" ? "#4CAF50" : type === "error" ? "#f44336" : "#2196F3";
            
            const $notif = $(`
                <div class="notification" style="
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: ${bgColor};
                    color: white;
                    padding: 15px 25px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    z-index: 10000;
                    animation: slideInRight 0.3s ease;
                ">
                    ${msg}
                </div>
            `);

            $('body').append($notif);

            setTimeout(() => {
                $notif.fadeOut(300, () => $notif.remove());
            }, 3000);
        }

        // -----------------------------------------------------------
        // CLUSTERS (graphique ML)
        // -----------------------------------------------------------
        setupClusterTabs() {
            const self = this;

            $(document).on("click", ".cluster-btn", async function (e) {
                e.preventDefault();
                const model = $(this).data("model") || "kmeans";
                self.currentClusterModel = model;

                $(".cluster-btn").removeClass("active");
                $(this).addClass("active");

                await self.loadClusterGraph(model);
            });
        }

        async loadClusterGraph(model = "kmeans") {
            let $container = $("#clusterResults");
            if (!$container.length) {
                $container = $(".chart-box");
            }

            if (!$container.length) {
                console.warn("Aucun conteneur de clusters trouvé.");
                return;
            }

            $container.html(`
                <div class="loading-clusters" style="text-align:center; padding:2rem;">
                    <div class="spinner" style="margin-bottom:1rem;"></div>
                    <p>Chargement des graphiques (${model})…</p>
                </div>
            `);

            try {
                const response = await $.ajax({
                    url: `/api/analytics/clusters?model=${encodeURIComponent(model)}`,
                    method: "GET"
                });

                if (response.error) {
                    $container.html(`
                        <div class="error-message">
                            <i class="icon-warning"></i>
                            <p>${response.error}</p>
                        </div>
                    `);
                    return;
                }

                // Afficher les 2 graphiques L'UN EN DESSOUS DE L'AUTRE pour qu'ils soient GÉANTS
                $container.html(`
                    <div class="cluster-results" style="animation: fadeInUp 0.6s ease forwards;">
                        <h4 style="text-align:center; margin-bottom:3rem; color:white; font-size:2rem;">
                            ${response.model_type || model} - Visualisations
                        </h4>
                        
                        <!-- GRAPH 1 - PLEINE LARGEUR -->
                        <div class="graph-container" style="margin-bottom: 4rem;">
                            <h5 style="text-align:center; margin-bottom:2rem; color:white; font-size:1.5rem;">
                                ${response.graph1.title}
                            </h5>
                            <img src="${response.graph1.image}"
                                 alt="${response.graph1.title}"
                                 style="width:100%; height:auto; border-radius:8px; box-shadow:0 10px 30px rgba(0,0,0,0.3);">
                        </div>
                        
                        <!-- GRAPH 2 - PLEINE LARGEUR -->
                        <div class="graph-container" style="margin-bottom: 2rem;">
                            <h5 style="text-align:center; margin-bottom:2rem; color:white; font-size:1.5rem;">
                                ${response.graph2.title}
                            </h5>
                            <img src="${response.graph2.image}"
                                 alt="${response.graph2.title}"
                                 style="width:100%; height:auto; border-radius:8px; box-shadow:0 10px 30px rgba(0,0,0,0.3);">
                        </div>
                        
                        <p style="text-align:center; color:#ccc; font-size:1.1rem; margin-top:2rem;">
                             ${response.message || "Graphiques chargés avec succès"}
                        </p>
                    </div>
                `);

            } catch (e) {
                console.error("Erreur graphique clusters:", e);
                $container.html(`
                    <div class="error-message">
                        <i class="icon-warning"></i>
                        <p>Erreur lors du chargement du graphique de clusters.</p>
                    </div>
                `);
            }
        }

        handleClusterClick(e) {
            const model = $(e.currentTarget).data('model');
            this.loadClusterGraph(model);
        }
    }

    // -----------------------------------------------------------
    // ANIMATION MANAGER
    // -----------------------------------------------------------
    class AnimationManager {
        constructor() {
            this.observeAnimations();
        }

        observeAnimations() {
            const obs = new IntersectionObserver((entries) => {
                entries.forEach(e => {
                    if (e.isIntersecting) {
                        e.target.classList.add("fade-in");
                    }
                });
            });

            document.querySelectorAll(".section, .patient-card").forEach(el => obs.observe(el));
        }
    }

    // Lancer l'app
    $(document).ready(() => {
        window.app = new MindCareApp();
    });

})(jQuery);