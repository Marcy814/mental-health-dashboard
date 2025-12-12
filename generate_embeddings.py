"""
Script pour générer et stocker les embeddings dans MongoDB
Selon les instructions du prof - Partie 2, Étape 1-2
"""

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import sys
import os

# Ajouter le chemin pour importer config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

def build_patient_description(patient):
    """
    Construit une description textuelle complète du patient
    pour générer l'embedding
    """
    parts = []
    
    # Informations démographiques
    if patient.get("Name"):
        parts.append(f"Nom: {patient['Name']}")
    if patient.get("Age"):
        parts.append(f"Âge: {patient['Age']} ans")
    if patient.get("Gender"):
        parts.append(f"Genre: {patient['Gender']}")
    if patient.get("Profession"):
        parts.append(f"Profession: {patient['Profession']}")
    
    # Situation familiale et sociale
    if patient.get("Marital Status"):
        parts.append(f"État civil: {patient['Marital Status']}")
    if patient.get("Number of Children"):
        parts.append(f"Nombre d'enfants: {patient['Number of Children']}")
    
    # Situation financière
    if patient.get("Income"):
        parts.append(f"Revenu: {patient['Income']} $")
    
    # Santé mentale
    if patient.get("History of Mental Illness"):
        parts.append(f"Historique de maladie mentale: {patient['History of Mental Illness']}")
    if patient.get("Family History of Depression"):
        parts.append(f"Historique familial de dépression: {patient['Family History of Depression']}")
    if patient.get("Chronic Medical Conditions"):
        parts.append(f"Conditions médicales chroniques: {patient['Chronic Medical Conditions']}")
    
    # Comportements
    if patient.get("Alcohol Consumption"):
        parts.append(f"Consommation d'alcool: {patient['Alcohol Consumption']}")
    if patient.get("Smoking Status"):
        parts.append(f"Statut fumeur: {patient['Smoking Status']}")
    if patient.get("Physical Activity Level"):
        parts.append(f"Niveau d'activité physique: {patient['Physical Activity Level']}")
    if patient.get("Sleep Disorder"):
        parts.append(f"Troubles du sommeil: {patient['Sleep Disorder']}")
    
    # Support social
    if patient.get("Social Support Rating"):
        parts.append(f"Niveau de support social: {patient['Social Support Rating']}/10")
    if patient.get("Stress Level"):
        parts.append(f"Niveau de stress: {patient['Stress Level']}/10")
    
    return ". ".join(parts)


def generate_and_store_embeddings(skip_existing=True):
    """
    Génère les embeddings pour TOUS les patients et les stocke dans MongoDB
    skip_existing: Si True, ne régénère pas les embeddings déjà existants
    """
    print("=" * 60)
    print("GÉNÉRATION ET STOCKAGE DES EMBEDDINGS")
    print("(TOUS LES PATIENTS - Traitement par lots)")
    print("=" * 60)
    
    # 1. Connexion MongoDB
    print("\n1. Connexion à MongoDB...")
    try:
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.DATABASE_NAME]
        collection = db[Config.COLLECTION_NAME]
        print(f"    Connecté à la collection: {Config.COLLECTION_NAME}")
    except Exception as e:
        print(f"   Erreur de connexion: {e}")
        return
    
    # 2. Charger le modèle d'embedding
    print("\n2. Chargement du modèle Sentence Transformer...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("    Modèle chargé (all-MiniLM-L6-v2)")
        print("    Dimension des vecteurs: 384")
    except Exception as e:
        print(f"    Erreur de chargement du modèle: {e}")
        return
    
    # 3. Récupérer les statistiques
    print("\n3. Analyse de la base de données...")
    try:
        total_patients = collection.count_documents({})
        print(f"    Total de patients dans la base: {total_patients}")
        
        # Vérifier combien ont déjà un embedding
        patients_with_embedding = collection.count_documents({"embedding": {"$exists": True}})
        print(f"    Patients avec embedding existant: {patients_with_embedding}")
        
        patients_to_process = total_patients - patients_with_embedding
        print(f"    Patients restant à traiter: {patients_to_process}")
        
        if patients_to_process == 0:
            print("\n    Tous les patients ont déjà un embedding!")
            response = input("   Voulez-vous régénérer TOUS les embeddings? (o/n): ")
            if response.lower() != 'o':
                print("    Opération annulée")
                return
            skip_existing = False
        
    except Exception as e:
        print(f"    Erreur: {e}")
        return
    
    # 4. Générer et stocker les embeddings PAR LOTS
    print("\n4. Génération des embeddings...")
    print("   (Traitement par lots de 100 pour éviter cursor timeout)")
    
    if skip_existing:
        print("   (Les embeddings existants seront conservés)\n")
    else:
        print("   (Tous les embeddings seront régénérés)\n")
    
    batch_size = 100
    success_count = patients_with_embedding if skip_existing else 0
    error_count = 0
    skipped_count = 0
    
    # Traiter par lots avec skip/limit
    for skip in range(0, total_patients, batch_size):
        try:
            # Récupérer un lot de patients
            if skip_existing:
                # Ne prendre que ceux SANS embedding
                batch = list(collection.find({"embedding": {"$exists": False}}).limit(batch_size))
            else:
                # Prendre tous les patients
                batch = list(collection.find({}).skip(skip).limit(batch_size))
            
            if not batch:
                break  # Plus de patients à traiter
            
            for patient in batch:
                try:
                    # Si skip_existing et a déjà un embedding, sauter
                    if skip_existing and "embedding" in patient:
                        skipped_count += 1
                        continue
                    
                    # Construire la description
                    description = build_patient_description(patient)
                    
                    # Générer l'embedding
                    embedding = model.encode(description).tolist()
                    
                    # Stocker dans MongoDB
                    collection.update_one(
                        {"_id": patient["_id"]},
                        {"$set": {"embedding": embedding}}
                    )
                    
                    success_count += 1
                    
                    # Afficher la progression
                    if success_count % 100 == 0:
                        progress = (success_count / total_patients * 100)
                        print(f"    {success_count}/{total_patients} patients traités ({progress:.1f}%)...")
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Afficher seulement les 5 premières erreurs
                        print(f"    Erreur pour patient {patient.get('Name', 'inconnu')}: {e}")
        
        except Exception as e:
            print(f"    Erreur lors du traitement du lot à partir de {skip}: {e}")
            continue
    
    # 5. Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f" Embeddings générés avec succès: {success_count}")
    if skipped_count > 0:
        print(f"⏭  Embeddings déjà existants (conservés): {skipped_count}")
    print(f" Erreurs: {error_count}")
    print(f" Total dans la base: {total_patients}")
    
    # Vérification finale
    final_count = collection.count_documents({"embedding": {"$exists": True}})
    print(f"\n Patients avec embedding dans la base: {final_count}/{total_patients}")
    
    if final_count == total_patients:
        print("\n PARFAIT! Tous les patients ont maintenant un embedding!")
    else:
        print(f"\n  Il manque encore {total_patients - final_count} embeddings")
        print("   Relancez le script pour compléter.")
    
    print("\n ÉTAPE 1-2 TERMINÉE: Embeddings stockés dans MongoDB!")
    print("\nProchaine étape: Créer l'index vectoriel dans MongoDB Atlas")
    print("=" * 60)


if __name__ == "__main__":
    # Traiter TOUS les patients (413,766)
    # skip_existing=True: conserve les embeddings déjà générés
    generate_and_store_embeddings(skip_existing=True)