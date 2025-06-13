import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class DataProcessor:
    def __init__(self):
        """NBA maç verisi işleme sınıfı"""
        self.feature_columns = None
        self.scaler = None
        self.load_feature_info()
    
    def load_feature_info(self):
        """Eğitim verisinden feature bilgilerini yükle"""
        try:
            with open('model/X_train_final_scaled.pkl', 'rb') as f:
                X_train = pickle.load(f)
                self.feature_columns = X_train.columns.tolist()
                print(f"✅ {len(self.feature_columns)} feature yüklendi")
        except Exception as e:
            print(f"❌ Feature bilgileri yüklenemedi: {e}")
    
    def create_team_features(self, home_team_stats, visitor_team_stats):
        """
        Takım istatistiklerinden model için gerekli feature'ları oluştur
        """
        features = {}
        
        # HOME TEAM FEATURES
        home_prefix = "HOME_"
        for key, value in home_team_stats.items():
            features[f"{home_prefix}{key}"] = value
        
        # VISITOR TEAM FEATURES  
        visitor_prefix = "VISITOR_"
        for key, value in visitor_team_stats.items():
            features[f"{visitor_prefix}{key}"] = value
        
        # DIFFERENCE FEATURES (HOME - VISITOR)
        for key in home_team_stats.keys():
            if key in visitor_team_stats:
                features[f"DIFF_{key}"] = home_team_stats[key] - visitor_team_stats[key]
        
        return features
    
    def prepare_input_data(self, home_team_stats, visitor_team_stats):
        """
        Kullanıcı girişlerini model input formatına dönüştür
        """
        # Feature'ları oluştur
        features = self.create_team_features(home_team_stats, visitor_team_stats)
        
        # DataFrame oluştur
        input_df = pd.DataFrame([features])
        
        # Feature columns kontrolü
        if self.feature_columns is None:
            print("⚠️ Feature columns yüklenmedi, temel feature'lar kullanılacak")
            # Temel feature'ları kullan
            return input_df
        
        # Orijinal modelde eksik olan feature'ları ekle
        missing_features = {
            'HOME_STAR_FORM_LAST_8': 0.0,
            'HOME_TS_PCT_DIFF': 0.0,
            'HOME_TS_PCT_LAST_8': 0.0,
            'HOME_WINS_PCT': home_team_stats.get('W_PCT', 0.0),
            'VISITOR_ROAD_WINS_PCT': visitor_team_stats.get('W_PCT', 0.0) * 0.8,  # Tahmini
            'VISITOR_STAR_FORM_LAST_8': 0.0,
            'VISITOR_TS_PCT_DIFF': 0.0,
            'VISITOR_TS_PCT_LAST_8': 0.0,
            'VISITOR_WINS_PCT': visitor_team_stats.get('W_PCT', 0.0),
            'HOME_ROAD_WINS_PCT': home_team_stats.get('W_PCT', 0.0) * 0.8,  # Tahmini
        }
        
        # Eksik feature'ları ekle
        for feature, value in missing_features.items():
            if feature not in input_df.columns:
                input_df[feature] = value
        
        # Eksik feature'ları 0 ile doldur
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Sadece model feature'larını seç
        try:
            input_df = input_df[self.feature_columns]
        except KeyError as e:
            print(f"⚠️ Eksik feature'lar: {e}")
            # Mevcut feature'ları kullan
            available_features = [col for col in self.feature_columns if col in input_df.columns]
            input_df = input_df[available_features]
            
            # Eksik feature'ları 0 ile ekle
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[self.feature_columns]
        
        return input_df
    
    def get_team_stats_template(self):
        """
        Kullanıcıdan alınacak takım istatistikleri şablonu
        """
        return {
            # Temel İstatistikler
            'PTS': 0.0,          # Points per game
            'FGM': 0.0,          # Field Goals Made
            'FGA': 0.0,          # Field Goals Attempted
            'FG_PCT': 0.0,       # Field Goal Percentage
            'FG3M': 0.0,         # 3-Point Field Goals Made
            'FG3A': 0.0,         # 3-Point Field Goals Attempted
            'FG3_PCT': 0.0,      # 3-Point Field Goal Percentage
            'FTM': 0.0,          # Free Throws Made
            'FTA': 0.0,          # Free Throws Attempted
            'FT_PCT': 0.0,       # Free Throw Percentage
            
            # Ribaund İstatistikleri
            'OREB': 0.0,         # Offensive Rebounds
            'DREB': 0.0,         # Defensive Rebounds
            'REB': 0.0,          # Total Rebounds
            
            # Diğer İstatistikler
            'AST': 0.0,          # Assists
            'STL': 0.0,          # Steals
            'BLK': 0.0,          # Blocks
            'TOV': 0.0,          # Turnovers
            'PF': 0.0,           # Personal Fouls
            
            # İleri İstatistikler
            'PLUS_MINUS': 0.0,   # Plus/Minus
            'W_PCT': 0.0,        # Win Percentage
            
            # Eksik feature'lar (hatadan görülen)
            'B2B': 0.0,          # Back to back games
            'DAYS_REST': 0.0,    # Days of rest
            'NETRTG_DIFF': 0.0,  # Net rating difference
            'NETRTG_LAST_8': 0.0, # Net rating last 8 games
            'RANK_W_PCT': 0.0,   # Win percentage rank
        }
    
    def validate_team_stats(self, team_stats):
        """
        Takım istatistiklerini doğrula
        """
        errors = []
        
        # Yüzde değerleri kontrol et
        percentage_fields = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'W_PCT']
        for field in percentage_fields:
            if field in team_stats:
                if not (0 <= team_stats[field] <= 1):
                    errors.append(f"{field} 0-1 arasında olmalı")
        
        # Negatif olamayacak değerler
        positive_fields = ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                          'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        for field in positive_fields:
            if field in team_stats:
                if team_stats[field] < 0:
                    errors.append(f"{field} negatif olamaz")
        
        # Mantık kontrolleri
        if 'FGM' in team_stats and 'FGA' in team_stats:
            if team_stats['FGM'] > team_stats['FGA']:
                errors.append("Field Goals Made, Field Goals Attempted'dan büyük olamaz")
        
        if 'FG3M' in team_stats and 'FG3A' in team_stats:
            if team_stats['FG3M'] > team_stats['FG3A']:
                errors.append("3-Point Made, 3-Point Attempted'dan büyük olamaz")
        
        if 'FTM' in team_stats and 'FTA' in team_stats:
            if team_stats['FTM'] > team_stats['FTA']:
                errors.append("Free Throws Made, Free Throws Attempted'dan büyük olamaz")
        
        return errors
    
    def get_feature_importance_info(self):
        """
        Model için en önemli feature'ları döndür
        """
        important_features = [
            'HOME_W_PCT', 'VISITOR_W_PCT', 'DIFF_W_PCT',
            'HOME_PTS', 'VISITOR_PTS', 'DIFF_PTS',
            'HOME_FG_PCT', 'VISITOR_FG_PCT', 'DIFF_FG_PCT',
            'HOME_PLUS_MINUS', 'VISITOR_PLUS_MINUS', 'DIFF_PLUS_MINUS'
        ]
        return important_features 