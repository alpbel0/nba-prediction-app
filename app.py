import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# CSS stilini yükle
st.markdown("""
<style>
.main { padding-top: 1rem; }
.team-header { 
    font-size: 24px; 
    font-weight: bold; 
    text-align: center; 
    padding: 20px; 
    border-radius: 10px; 
    margin-bottom: 20px;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    color: white;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 5px 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.formula-box {
    background-color: #e3f2fd;
    padding: 8px;
    border-radius: 5px;
    margin: 5px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

class HybridNBAPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        """Gerçek eğitilmiş model ve scaler'ı yükle"""
        try:
            base_path = os.path.dirname(__file__)
            model_path = os.path.join(base_path, "model", "best_model_logistic_regression.pkl") 
            scaler_path = os.path.join(base_path, "model", "compatible_scaler.pkl") 

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                
                # Scaler yükleme - hata varsa fallback
                try:
                    if os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        st.success("✅ Model ve scaler başarıyla yüklendi!")
                    else:
                        st.warning("⚠️ Scaler dosyası bulunamadı - varsayılan scaler kullanılıyor")
                        self.scaler = StandardScaler()
                        np.random.seed(42)
                        X_demo = np.random.uniform(-1, 1, (100, 17))
                        self.scaler.fit(X_demo)
                except Exception as scaler_error:
                    st.warning(f"⚠️ Scaler yükleme hatası: {scaler_error} - Varsayılan scaler kullanılıyor")
                    self.scaler = StandardScaler()
                    np.random.seed(42)
                    X_demo = np.random.uniform(-1, 1, (100, 17))
                    self.scaler.fit(X_demo)
                
                self.feature_names = [
                    'HOME_RANK_W_PCT', 'VISITOR_RANK_W_PCT', 'HOME_WINS_PCT', 'VISITOR_ROAD_WINS_PCT', 
                    'HOME_DAYS_REST', 'HOME_B2B', 'VISITOR_B2B', 'HOME_TS_PCT_LAST_8', 
                    'HOME_TS_PCT_DIFF', 'HOME_NETRTG_LAST_8', 'HOME_NETRTG_DIFF', 'VISITOR_TS_PCT_LAST_8', 
                    'VISITOR_TS_PCT_DIFF', 'VISITOR_NETRTG_LAST_8', 'VISITOR_NETRTG_DIFF', 
                    'HOME_STAR_FORM_LAST_8', 'VISITOR_STAR_FORM_LAST_8'
                ]
            else:
                st.error(f"❌ Model dosyası bulunamadı! Lütfen '{model_path}' dosyasını kontrol edin.")
                self.model = LogisticRegression(random_state=42)
                self.scaler = StandardScaler()
                np.random.seed(42)
                X_demo = np.random.uniform(-1, 1, (100, 17))
                y_demo = np.random.randint(0, 2, 100)
                self.model.fit(X_demo, y_demo)
                self.feature_names = [f'feature_{i}' for i in range(17)]
                st.warning("⚠️ Gerçek model bulunamadı - Demo model kullanılıyor.")
                
        except Exception as e:
            st.error(f"❌ Model yüklenemedi: {e}")
            self.model = None
            self.scaler = None
    
    def calculate_ts_pct(self, pts, fga, fta):
        """True Shooting % hesapla"""
        if fga == 0 and fta == 0:
            return 0.5  # Varsayılan değer
        return pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0.5
    
    def calculate_star_form(self, pts, reb, ast, fg_pct, mins):
        """Yıldız oyuncu formu hesapla"""
        return pts + (reb * 1.2) + (ast * 1.5) + (fg_pct * 30) + (mins * 0.5)
    
    def calculate_net_rating(self, ortg, drtg):
        """Net Rating hesapla"""
        return ortg - drtg
    
    def predict_game(self, home_stats, visitor_stats):
        """Ham istatistiklerden feature'ları hesaplayıp tahmin yap"""
        try:
            if self.model is None or self.scaler is None:
                return {"error": "Model veya scaler yüklenmemiş"}
            
            # Ham verilerden feature'ları hesapla
            features = {}
            
            # Ev sahibi takım özellikleri
            features['HOME_RANK_W_PCT'] = home_stats['wins'] / max(home_stats['games_played'], 1)
            features['HOME_WINS_PCT'] = home_stats['home_wins'] / max(home_stats['home_games'], 1)
            features['HOME_DAYS_REST'] = home_stats['days_rest']
            features['HOME_B2B'] = 1 if home_stats['back_to_back'] else 0
            
            # True Shooting % hesapla (son 8 maç)
            home_ts_last8 = self.calculate_ts_pct(
                home_stats['pts_last8'], home_stats['fga_last8'], home_stats['fta_last8']
            )
            home_ts_season = self.calculate_ts_pct(
                home_stats['pts_season'], home_stats['fga_season'], home_stats['fta_season']
            )
            features['HOME_TS_PCT_LAST_8'] = home_ts_last8
            features['HOME_TS_PCT_DIFF'] = home_ts_last8 - home_ts_season
            
            # Net Rating hesapla (son 8 maç)
            home_netrtg_last8 = self.calculate_net_rating(home_stats['ortg_last8'], home_stats['drtg_last8'])
            home_netrtg_season = self.calculate_net_rating(home_stats['ortg_season'], home_stats['drtg_season'])
            features['HOME_NETRTG_LAST_8'] = home_netrtg_last8
            features['HOME_NETRTG_DIFF'] = home_netrtg_last8 - home_netrtg_season
            
            # Yıldız oyuncu formu (son 8 maç)
            features['HOME_STAR_FORM_LAST_8'] = self.calculate_star_form(
                home_stats['star_pts_last8'], home_stats['star_reb_last8'], 
                home_stats['star_ast_last8'], home_stats['star_fg_pct_last8'],
                home_stats['star_min_last8']
            )
            
            # Misafir takım özellikleri
            features['VISITOR_RANK_W_PCT'] = visitor_stats['wins'] / max(visitor_stats['games_played'], 1)
            features['VISITOR_ROAD_WINS_PCT'] = visitor_stats['road_wins'] / max(visitor_stats['road_games'], 1)
            features['VISITOR_B2B'] = 1 if visitor_stats['back_to_back'] else 0
            
            # True Shooting % hesapla (son 8 maç)
            visitor_ts_last8 = self.calculate_ts_pct(
                visitor_stats['pts_last8'], visitor_stats['fga_last8'], visitor_stats['fta_last8']
            )
            visitor_ts_season = self.calculate_ts_pct(
                visitor_stats['pts_season'], visitor_stats['fga_season'], visitor_stats['fta_season']
            )
            features['VISITOR_TS_PCT_LAST_8'] = visitor_ts_last8
            features['VISITOR_TS_PCT_DIFF'] = visitor_ts_last8 - visitor_ts_season
            
            # Net Rating hesapla (son 8 maç)
            visitor_netrtg_last8 = self.calculate_net_rating(visitor_stats['ortg_last8'], visitor_stats['drtg_last8'])
            visitor_netrtg_season = self.calculate_net_rating(visitor_stats['ortg_season'], visitor_stats['drtg_season'])
            features['VISITOR_NETRTG_LAST_8'] = visitor_netrtg_last8
            features['VISITOR_NETRTG_DIFF'] = visitor_netrtg_last8 - visitor_netrtg_season
            
            # Yıldız oyuncu formu (son 8 maç)
            features['VISITOR_STAR_FORM_LAST_8'] = self.calculate_star_form(
                visitor_stats['star_pts_last8'], visitor_stats['star_reb_last8'], 
                visitor_stats['star_ast_last8'], visitor_stats['star_fg_pct_last8'],
                visitor_stats['star_min_last8']
            )
            
            # Model tahminini yap
            model_input_df = pd.DataFrame([features], columns=self.feature_names)
            X_scaled = self.scaler.transform(model_input_df)
            
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            home_win_prob = probabilities[1]
            visitor_win_prob = probabilities[0]
            
            winner = 'HOME' if home_win_prob > visitor_win_prob else 'VISITOR'
            
            max_prob = max(home_win_prob, visitor_win_prob)
            if max_prob >= 0.75:
                confidence = 'YÜKSEK'
            elif max_prob >= 0.65:
                confidence = 'ORTA'
            else:
                confidence = 'DÜŞÜK'
            
            return {
                'winner': winner,
                'home_win_probability': float(home_win_prob),
                'visitor_win_probability': float(visitor_win_prob),
                'confidence': confidence,
                'calculated_features': features
            }
            
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}

def main():
    st.title("🏀 NBA Maç Tahmin Sistemi")
    
    st.info("""
    💡 **Nasıl Kullanılır:** Takım isimlerini girin ve basketbol istatistiklerini doldurun. 
    Sistem otomatik olarak gerekli hesaplamaları yaparak tahmin üretecek.
    """)
    
    st.markdown("---")
    
    predictor = HybridNBAPredictor()
    
    if predictor.model is None:
        st.error("Model yüklenemedi. Lütfen model dosyalarını kontrol edin.")
        return
    
    # İki sütun layout
    col1, col2 = st.columns(2)
    
    # Ev Sahibi Takım
    with col1:
        st.markdown('<div class="team-header">🏠 Ev Sahibi Takım</div>', unsafe_allow_html=True)
        home_team_name = st.text_input("Takım Adı", value="Lakers", key="home_name")
        
        st.subheader("📊 Genel Takım İstatistikleri")
        home_games_played = st.number_input("Toplam Oynadığı Maç", min_value=1, max_value=82, value=50, key="home_games")
        home_wins = st.number_input("Toplam Galibiyet", min_value=0, max_value=82, value=30, key="home_wins")
        home_games_home = st.number_input("İç Sahada Oynadığı Maç", min_value=1, max_value=41, value=25, key="home_games_home")
        home_wins_home = st.number_input("İç Sahada Galibiyet", min_value=0, max_value=41, value=18, key="home_wins_home")
        home_days_rest = st.number_input("Son Maçtan Bu Yana Dinlenme Günü", min_value=0, max_value=10, value=2, key="home_rest")
        home_b2b = st.checkbox("Dün de Maç Oynadı (Back-to-Back)", key="home_b2b")
        
        st.subheader("🏀 Sezon Ortalamaları")
        home_pts_season = st.number_input("Sezon Ortalama Sayı", min_value=80.0, max_value=130.0, value=110.0, step=0.1, key="home_pts_season")
        home_fga_season = st.number_input("Sezon Ortalama Şut Denemesi", min_value=70.0, max_value=110.0, value=85.0, step=0.1, key="home_fga_season")
        home_fta_season = st.number_input("Sezon Ortalama Serbest Atış", min_value=10.0, max_value=35.0, value=20.0, step=0.1, key="home_fta_season")
        home_ortg_season = st.number_input("Sezon Hücum Rating (ORTG)", min_value=90.0, max_value=130.0, value=110.0, step=0.1, key="home_ortg_season")
        home_drtg_season = st.number_input("Sezon Savunma Rating (DRTG)", min_value=90.0, max_value=130.0, value=108.0, step=0.1, key="home_drtg_season")
        
        st.subheader("⚡ Son 8 Maç Ortalamaları")
        home_pts_last8 = st.number_input("Son 8 Maç Ortalama Sayı", min_value=80.0, max_value=130.0, value=112.0, step=0.1, key="home_pts_last8")
        home_fga_last8 = st.number_input("Son 8 Maç Ortalama Şut Denemesi", min_value=70.0, max_value=110.0, value=87.0, step=0.1, key="home_fga_last8")
        home_fta_last8 = st.number_input("Son 8 Maç Ortalama Serbest Atış", min_value=10.0, max_value=35.0, value=22.0, step=0.1, key="home_fta_last8")
        home_ortg_last8 = st.number_input("Son 8 Maç Hücum Rating (ORTG)", min_value=90.0, max_value=130.0, value=112.0, step=0.1, key="home_ortg_last8")
        home_drtg_last8 = st.number_input("Son 8 Maç Savunma Rating (DRTG)", min_value=90.0, max_value=130.0, value=106.0, step=0.1, key="home_drtg_last8")
        
        st.subheader("⭐ Yıldız Oyuncu (Son 8 Maç)")
        home_star_pts = st.number_input("Yıldız Oyuncu Ortalama Sayı", min_value=15.0, max_value=40.0, value=25.0, step=0.1, key="home_star_pts")
        home_star_reb = st.number_input("Yıldız Oyuncu Ortalama Ribaund", min_value=3.0, max_value=15.0, value=8.0, step=0.1, key="home_star_reb")
        home_star_ast = st.number_input("Yıldız Oyuncu Ortalama Asist", min_value=2.0, max_value=12.0, value=6.0, step=0.1, key="home_star_ast")
        home_star_fg_pct = st.number_input("Yıldız Oyuncu Şut %", min_value=0.3, max_value=0.7, value=0.45, step=0.01, key="home_star_fg")
        home_star_min = st.number_input("Yıldız Oyuncu Ortalama Dakika", min_value=25.0, max_value=42.0, value=35.0, step=0.1, key="home_star_min")
        
    # Misafir Takım  
    with col2:
        st.markdown('<div class="team-header">✈️ Misafir Takım</div>', unsafe_allow_html=True)
        visitor_team_name = st.text_input("Takım Adı", value="Warriors", key="visitor_name")
        
        st.subheader("📊 Genel Takım İstatistikleri")
        visitor_games_played = st.number_input("Toplam Oynadığı Maç", min_value=1, max_value=82, value=48, key="visitor_games")
        visitor_wins = st.number_input("Toplam Galibiyet", min_value=0, max_value=82, value=28, key="visitor_wins")
        visitor_games_road = st.number_input("Deplasmanda Oynadığı Maç", min_value=1, max_value=41, value=24, key="visitor_games_road")
        visitor_wins_road = st.number_input("Deplasmanda Galibiyet", min_value=0, max_value=41, value=12, key="visitor_wins_road")
        visitor_b2b = st.checkbox("Dün de Maç Oynadı (Back-to-Back)", key="visitor_b2b")
        
        st.subheader("🏀 Sezon Ortalamaları")
        visitor_pts_season = st.number_input("Sezon Ortalama Sayı", min_value=80.0, max_value=130.0, value=108.0, step=0.1, key="visitor_pts_season")
        visitor_fga_season = st.number_input("Sezon Ortalama Şut Denemesi", min_value=70.0, max_value=110.0, value=83.0, step=0.1, key="visitor_fga_season")
        visitor_fta_season = st.number_input("Sezon Ortalama Serbest Atış", min_value=10.0, max_value=35.0, value=18.0, step=0.1, key="visitor_fta_season")
        visitor_ortg_season = st.number_input("Sezon Hücum Rating (ORTG)", min_value=90.0, max_value=130.0, value=108.0, step=0.1, key="visitor_ortg_season")
        visitor_drtg_season = st.number_input("Sezon Savunma Rating (DRTG)", min_value=90.0, max_value=130.0, value=110.0, step=0.1, key="visitor_drtg_season")
        
        st.subheader("⚡ Son 8 Maç Ortalamaları")
        visitor_pts_last8 = st.number_input("Son 8 Maç Ortalama Sayı", min_value=80.0, max_value=130.0, value=109.0, step=0.1, key="visitor_pts_last8")
        visitor_fga_last8 = st.number_input("Son 8 Maç Ortalama Şut Denemesi", min_value=70.0, max_value=110.0, value=84.0, step=0.1, key="visitor_fga_last8")
        visitor_fta_last8 = st.number_input("Son 8 Maç Ortalama Serbest Atış", min_value=10.0, max_value=35.0, value=19.0, step=0.1, key="visitor_fta_last8")
        visitor_ortg_last8 = st.number_input("Son 8 Maç Hücum Rating (ORTG)", min_value=90.0, max_value=130.0, value=109.0, step=0.1, key="visitor_ortg_last8")
        visitor_drtg_last8 = st.number_input("Son 8 Maç Savunma Rating (DRTG)", min_value=90.0, max_value=130.0, value=108.0, step=0.1, key="visitor_drtg_last8")
        
        st.subheader("⭐ Yıldız Oyuncu (Son 8 Maç)")
        visitor_star_pts = st.number_input("Yıldız Oyuncu Ortalama Sayı", min_value=15.0, max_value=40.0, value=24.0, step=0.1, key="visitor_star_pts")
        visitor_star_reb = st.number_input("Yıldız Oyuncu Ortalama Ribaund", min_value=3.0, max_value=15.0, value=7.0, step=0.1, key="visitor_star_reb")
        visitor_star_ast = st.number_input("Yıldız Oyuncu Ortalama Asist", min_value=2.0, max_value=12.0, value=5.5, step=0.1, key="visitor_star_ast")
        visitor_star_fg_pct = st.number_input("Yıldız Oyuncu Şut %", min_value=0.3, max_value=0.7, value=0.43, step=0.01, key="visitor_star_fg")
        visitor_star_min = st.number_input("Yıldız Oyuncu Ortalama Dakika", min_value=25.0, max_value=42.0, value=34.0, step=0.1, key="visitor_star_min")

    # Tahmin butonu
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 TAHMİN ET", use_container_width=True, type="primary")
    
    if predict_button:
        # Ham istatistikleri topla
        home_stats = {
            'games_played': home_games_played, 'wins': home_wins,
            'home_games': home_games_home, 'home_wins': home_wins_home,
            'days_rest': home_days_rest, 'back_to_back': home_b2b,
            'pts_season': home_pts_season, 'fga_season': home_fga_season, 'fta_season': home_fta_season,
            'ortg_season': home_ortg_season, 'drtg_season': home_drtg_season,
            'pts_last8': home_pts_last8, 'fga_last8': home_fga_last8, 'fta_last8': home_fta_last8,
            'ortg_last8': home_ortg_last8, 'drtg_last8': home_drtg_last8,
            'star_pts_last8': home_star_pts, 'star_reb_last8': home_star_reb,
            'star_ast_last8': home_star_ast, 'star_fg_pct_last8': home_star_fg_pct,
            'star_min_last8': home_star_min
        }
        
        visitor_stats = {
            'games_played': visitor_games_played, 'wins': visitor_wins,
            'road_games': visitor_games_road, 'road_wins': visitor_wins_road,
            'back_to_back': visitor_b2b,
            'pts_season': visitor_pts_season, 'fga_season': visitor_fga_season, 'fta_season': visitor_fta_season,
            'ortg_season': visitor_ortg_season, 'drtg_season': visitor_drtg_season,
            'pts_last8': visitor_pts_last8, 'fga_last8': visitor_fga_last8, 'fta_last8': visitor_fta_last8,
            'ortg_last8': visitor_ortg_last8, 'drtg_last8': visitor_drtg_last8,
            'star_pts_last8': visitor_star_pts, 'star_reb_last8': visitor_star_reb,
            'star_ast_last8': visitor_star_ast, 'star_fg_pct_last8': visitor_star_fg_pct,
            'star_min_last8': visitor_star_min
        }
        
        # Tahmin yap
        prediction = predictor.predict_game(home_stats, visitor_stats)
        
        if 'error' in prediction:
            st.error(f"Hata: {prediction['error']}")
        else:
            # Sonuçları göster
            st.markdown("## 🎯 Tahmin Sonucu")
            
            col_res1, col_res2 = st.columns(2)
            
            winner = prediction['winner']
            home_prob = prediction['home_win_probability']
            visitor_prob = prediction['visitor_win_probability']
            confidence = prediction['confidence']
            
            with col_res1:
                if winner == 'HOME':
                    st.success(f"🏠 **{home_team_name}** kazanacak!")
                    st.metric("Kazanma Olasılığı", f"{home_prob:.1%}")
                else:
                    st.success(f"✈️ **{visitor_team_name}** kazanacak!")
                    st.metric("Kazanma Olasılığı", f"{visitor_prob:.1%}")
                
                st.metric("Güven Seviyesi", confidence)
                
                # Grafik
                fig = go.Figure(data=[
                    go.Bar(name='Takımlar', 
                          x=[home_team_name, visitor_team_name],
                          y=[home_prob*100, visitor_prob*100],
                          marker_color=['#1f77b4', '#ff7f0e'])
                ])
                fig.update_layout(
                    title="Kazanma Olasılıkları (%)",
                    yaxis_title="Olasılık (%)",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_res2:
                st.markdown("### 🧮 Hesaplanan Model Özellikleri")
                features = prediction['calculated_features']
                
                st.markdown(f"**🏠 {home_team_name}:**")
                st.write(f"• Genel Kazanma %: {features['HOME_RANK_W_PCT']:.3f}")
                st.write(f"• Evde Kazanma %: {features['HOME_WINS_PCT']:.3f}")
                st.write(f"• True Shooting % (Son 8): {features['HOME_TS_PCT_LAST_8']:.3f}")
                st.write(f"• TS% Farkı: {features['HOME_TS_PCT_DIFF']:.3f}")
                st.write(f"• Net Rating (Son 8): {features['HOME_NETRTG_LAST_8']:.1f}")
                st.write(f"• Net Rating Farkı: {features['HOME_NETRTG_DIFF']:.1f}")
                st.write(f"• Yıldız Formu: {features['HOME_STAR_FORM_LAST_8']:.1f}")
                
                st.markdown(f"**✈️ {visitor_team_name}:**")
                st.write(f"• Genel Kazanma %: {features['VISITOR_RANK_W_PCT']:.3f}")
                st.write(f"• Deplasmanda Kazanma %: {features['VISITOR_ROAD_WINS_PCT']:.3f}")
                st.write(f"• True Shooting % (Son 8): {features['VISITOR_TS_PCT_LAST_8']:.3f}")
                st.write(f"• TS% Farkı: {features['VISITOR_TS_PCT_DIFF']:.3f}")
                st.write(f"• Net Rating (Son 8): {features['VISITOR_NETRTG_LAST_8']:.1f}")
                st.write(f"• Net Rating Farkı: {features['VISITOR_NETRTG_DIFF']:.1f}")
                st.write(f"• Yıldız Formu: {features['VISITOR_STAR_FORM_LAST_8']:.1f}")
            
            # Formül açıklamaları
            with st.expander("🧮 Kullanılan Formüller"):
                st.markdown("""
                **⚡ True Shooting %:**
                ```
                TS% = Sayı / (2 × (Şut Denemesi + 0.44 × Serbest Atış))
                ```
                
                **🎯 Net Rating:**
                ```
                Net Rating = Hücum Rating (ORTG) - Savunma Rating (DRTG)
                ```
                
                **⭐ Yıldız Formu:**
                ```
                Form = Sayı + (Ribaund × 1.2) + (Asist × 1.5) + (Şut% × 30) + (Dakika × 0.5)
                ```
                """)

if __name__ == "__main__":
    main() 