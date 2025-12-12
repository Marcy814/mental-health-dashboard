# Modèle de données pour les patients
from datetime import datetime
from pymongo import MongoClient
from config import Config


class PatientModel:
    def __init__(self):
        self.collection = self._get_collection()

    def _get_collection(self):
        """Établit la connexion à la collection patients"""
        try:
            client = MongoClient(Config.MONGO_URI)
            database = client[Config.DATABASE_NAME]
            return database[Config.COLLECTION_NAME]
        except Exception as e:
            print(f"Erreur connexion DB: {e}")
            return None

    def validate_patient_data(self, patient_data):
        """Valide les données du patient pour la base Depression"""
        # Les champs peuvent varier, ajustez selon votre structure
        expected_fields = ['Name', 'Age', 'Marital Status', 'Employment Status']

        missing_fields = [field for field in expected_fields if field not in patient_data]
        if missing_fields:
            return False, f"Champs manquants: {', '.join(missing_fields)}"

        # Validation de l'âge
        if not isinstance(patient_data['Age'], (int, float)) or patient_data['Age'] < 0:
            return False, "Âge invalide"

        return True, "Données valides"
    def create_patient(self, patient_data):
        """Crée un nouveau patient"""
        try:
            is_valid, message = self.validate_patient_data(patient_data)
            if not is_valid:
                return None, message

            # Ajout de métadonnées
            patient_data['created_at'] = datetime.utcnow()
            patient_data['last_updated'] = datetime.utcnow()

            result = self.collection.insert_one(patient_data)
            return result.inserted_id, "Patient créé avec succès"

        except Exception as e:
            return None, f"Erreur création patient: {str(e)}"

    def get_patient(self, patient_id):
        """Récupère un patient par son ID"""
        try:
            from bson import ObjectId
            patient = self.collection.find_one({'_id': ObjectId(patient_id)})
            return patient, "Patient trouvé" if patient else "Patient non trouvé"
        except Exception as e:
            return None, f"Erreur récupération patient: {str(e)}"

    def update_patient(self, patient_id, update_data):
        """Met à jour les données d'un patient"""
        try:
            from bson import ObjectId
            update_data['last_updated'] = datetime.utcnow()

            result = self.collection.update_one(
                {'_id': ObjectId(patient_id)},
                {'$set': update_data}
            )

            if result.modified_count > 0:
                return True, "Patient mis à jour avec succès"
            else:
                return False, "Aucune modification effectuée"

        except Exception as e:
            return False, f"Erreur mise à jour patient: {str(e)}"

    def delete_patient(self, patient_id):
        """Supprime un patient"""
        try:
            from bson import ObjectId
            result = self.collection.delete_one({'_id': ObjectId(patient_id)})

            if result.deleted_count > 0:
                return True, "Patient supprimé avec succès"
            else:
                return False, "Patient non trouvé"

        except Exception as e:
            return False, f"Erreur suppression patient: {str(e)}"

    def search_patients(self, search_criteria):
        """Recherche des patients selon des critères"""
        try:
            patients = list(self.collection.find(search_criteria))
            return patients, f"{len(patients)} patient(s) trouvé(s)"
        except Exception as e:
            return [], f"Erreur recherche: {str(e)}"