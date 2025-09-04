# 🏀 NBA Maç Tahmin Sistemi

Bu uygulama, NBA maçlarının sonuçlarını tahmin etmek için makine öğrenmesi modeli kullanır.

## 🚀 Kurulum

1. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Uygulamayı çalıştırın:**
```bash
streamlit run app.py
```

## 📊 Nasıl Kullanılır

### Gerekli Veriler:

**Her takım için:**
- Takım adı
- Genel maç istatistikleri (toplam maç, galibiyet)
- İç saha/deplasman performansı  
- Dinlenme durumu (back-to-back maç)
- **Sezon ortalamaları:** sayı, şut denemesi, serbest atış, ORTG, DRTG
- **Son 8 maç ortalamaları:** sayı, şut denemesi, serbest atış, ORTG, DRTG
- **Yıldız oyuncu stats:** sayı, ribaund, asist, şut%, dakika

### Sistem Otomatik Hesaplıyor:
- **True Shooting %:** TS% = Sayı / (2 × (Şut + 0.44 × Serbest Atış))
- **Net Rating:** ORTG - DRTG  
- **Yıldız Formu:** Sayı + (Rib×1.2) + (Ast×1.5) + (Şut%×30) + (Dak×0.5)

## 📁 Dosya Yapısı

```
streamlit_app/
├── app.py                    # Ana uygulama
├── requirements.txt          # Kütüphaneler  
├── model/
│   ├── best_model_logistic_regression.pkl  # Eğitilmiş model
│   └── compatible_scaler.pkl               # Scaler
└── README.md                # Bu dosya
```

## ✅ Özellikler

- ✅ Gerçek eğitilmiş model kullanır
- ✅ Ham basketbol verilerini otomatik işler
- ✅ Kullanıcı dostu arayüz
- ✅ Görsel tahmin sonuçları
- ✅ Hesaplama detayları
- ✅ Formül açıklamaları

## 🔧 Teknik Detaylar

- **Model:** Logistic Regression  
- **Features:** 17 özellik (kazanma %, TS%, Net Rating, vs.)
- **Framework:** Streamlit
- **ML:** scikit-learn, joblib 