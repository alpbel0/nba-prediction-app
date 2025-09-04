import pickle
import pandas as pd

# Orijinal feature'ları yükle
with open('Final/X_train_final_scaled.pkl', 'rb') as f:
    X_train = pickle.load(f)

print(f"Toplam feature sayısı: {len(X_train.columns)}")
print("\nİlk 30 feature:")
for i, col in enumerate(X_train.columns[:30], 1):
    print(f"{i:2d}. {col}")

print("\nHOME_ ile başlayan feature'lar:")
home_features = [col for col in X_train.columns if col.startswith('HOME_')]
print(f"HOME_ feature sayısı: {len(home_features)}")
for i, col in enumerate(home_features[:10], 1):
    print(f"{i:2d}. {col}")

print("\nVISITOR_ ile başlayan feature'lar:")
visitor_features = [col for col in X_train.columns if col.startswith('VISITOR_')]
print(f"VISITOR_ feature sayısı: {len(visitor_features)}")
for i, col in enumerate(visitor_features[:10], 1):
    print(f"{i:2d}. {col}")

print("\nDIFF_ ile başlayan feature'lar:")
diff_features = [col for col in X_train.columns if col.startswith('DIFF_')]
print(f"DIFF_ feature sayısı: {len(diff_features)}")
for i, col in enumerate(diff_features[:10], 1):
    print(f"{i:2d}. {col}") 