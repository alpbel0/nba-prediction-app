import pickle
import numpy as np
import pandas as pd
from .data_processor import DataProcessor

class NBAPredictor:
    def __init__(self):
        """NBA maç sonucu tahmin sınıfı"""
        self.model = None
        self.model_info = None
        self.data_processor = DataProcessor()
        self.load_model()
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            # Model dosyasını yükle
            with open('model/best_model_logistic_regression.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Model bilgilerini yükle
            with open('model/model_results_logistic_regression.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            
            print("✅ Model başarıyla yüklendi")
            print(f"📊 Model Accuracy: {self.model_info.get('test_accuracy', 'N/A'):.4f}")
            print(f"📈 Model AUC: {self.model_info.get('test_auc', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"❌ Model yüklenemedi: {e}")
    
    def predict_game(self, home_team_stats, visitor_team_stats):
        """
        Maç sonucunu tahmin et
        
        Returns:
            dict: {
                'winner': 'HOME' or 'VISITOR',
                'home_win_probability': float,
                'visitor_win_probability': float,
                'confidence': 'HIGH', 'MEDIUM', 'LOW'
            }
        """
        if self.model is None:
            return {"error": "Model yüklenmedi"}
        
        try:
            # Veriyi hazırla
            input_data = self.data_processor.prepare_input_data(
                home_team_stats, visitor_team_stats
            )
            
            # Tahmin yap
            prediction = self.model.predict(input_data)[0]
            probabilities = self.model.predict_proba(input_data)[0]
            
            home_win_prob = probabilities[1]  # Class 1 = HOME WIN
            visitor_win_prob = probabilities[0]  # Class 0 = VISITOR WIN
            
            # Kazanan takımı belirle
            winner = 'HOME' if prediction == 1 else 'VISITOR'
            
            # Güven seviyesini belirle
            max_prob = max(home_win_prob, visitor_win_prob)
            if max_prob >= 0.7:
                confidence = 'HIGH'
            elif max_prob >= 0.6:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'winner': winner,
                'home_win_probability': float(home_win_prob),
                'visitor_win_probability': float(visitor_win_prob),
                'confidence': confidence,
                'prediction_raw': int(prediction)
            }
            
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}
    
    def get_prediction_explanation(self, home_team_stats, visitor_team_stats):
        """
        Tahmin açıklaması ve önemli faktörler
        """
        try:
            # Veriyi hazırla
            input_data = self.data_processor.prepare_input_data(
                home_team_stats, visitor_team_stats
            )
            
            # Feature importance bilgilerini al
            if hasattr(self.model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'Feature': input_data.columns,
                    'Coefficient': self.model.coef_[0],
                    'Value': input_data.iloc[0].values,
                    'Impact': self.model.coef_[0] * input_data.iloc[0].values
                })
                
                # En etkili feature'ları bul
                feature_importance['Abs_Impact'] = np.abs(feature_importance['Impact'])
                top_features = feature_importance.nlargest(10, 'Abs_Impact')
                
                return {
                    'top_factors': top_features.to_dict('records'),
                    'home_advantages': top_features[top_features['Impact'] > 0]['Feature'].tolist(),
                    'visitor_advantages': top_features[top_features['Impact'] < 0]['Feature'].tolist()
                }
            else:
                return {"error": "Model feature importance desteklemiyor"}
                
        except Exception as e:
            return {"error": f"Açıklama hatası: {str(e)}"}
    
    def get_model_info(self):
        """Model performans bilgilerini döndür"""
        if self.model_info:
            return {
                'model_name': self.model_info.get('best_name', 'Logistic Regression'),
                'accuracy': self.model_info.get('test_accuracy', 0),
                'auc': self.model_info.get('test_auc', 0),
                'cv_mean': self.model_info.get('cv_mean', 0),
                'improvement_vs_baseline': {
                    'accuracy': self.model_info.get('acc_improvement', 0),
                    'auc': self.model_info.get('auc_improvement', 0)
                }
            }
        return {}
    
    def validate_prediction_input(self, home_team_stats, visitor_team_stats):
        """
        Tahmin girişlerini doğrula
        """
        errors = []
        
        # Home team doğrulama
        home_errors = self.data_processor.validate_team_stats(home_team_stats)
        errors.extend([f"Ev Sahibi - {error}" for error in home_errors])
        
        # Visitor team doğrulama
        visitor_errors = self.data_processor.validate_team_stats(visitor_team_stats)
        errors.extend([f"Misafir - {error}" for error in visitor_errors])
        
        return errors
    
    def get_team_comparison(self, home_team_stats, visitor_team_stats):
        """
        İki takımın karşılaştırmalı analizi
        """
        comparison = {}
        
        # Temel istatistik karşılaştırmaları
        key_stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'W_PCT']
        
        for stat in key_stats:
            if stat in home_team_stats and stat in visitor_team_stats:
                home_val = home_team_stats[stat]
                visitor_val = visitor_team_stats[stat]
                
                if home_val > visitor_val:
                    advantage = 'HOME'
                    difference = home_val - visitor_val
                elif visitor_val > home_val:
                    advantage = 'VISITOR'
                    difference = visitor_val - home_val
                else:
                    advantage = 'EQUAL'
                    difference = 0
                
                comparison[stat] = {
                    'home_value': home_val,
                    'visitor_value': visitor_val,
                    'advantage': advantage,
                    'difference': difference,
                    'difference_pct': (difference / max(home_val, visitor_val, 0.001)) * 100
                }
        
        return comparison
    
    def get_confidence_explanation(self, confidence, probability):
        """
        Güven seviyesi açıklaması
        """
        explanations = {
            'HIGH': f"Yüksek güven (%{probability*100:.1f}) - Model bu tahmin konusunda çok emin",
            'MEDIUM': f"Orta güven (%{probability*100:.1f}) - Model makul bir güvenle tahmin yapıyor",
            'LOW': f"Düşük güven (%{probability*100:.1f}) - Maç çok yakın, her iki takımın da şansı var"
        }
        return explanations.get(confidence, "Bilinmeyen güven seviyesi") 