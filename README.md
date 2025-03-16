# Simple OR Gate Neural Network

Bu proje, basit bir yapay sinir ağı kullanarak OR kapısını öğrenmeyi amaçlayan bir Python uygulamasıdır. Ağı, doğrusal olmayan ilişkileri öğrenme ve tahminler yapma yeteneğine sahip olan Sigmoid aktivasyon fonksiyonu kullanarak eğitmektedir.

## Kullanılan Kütüphaneler

- `numpy`: Sayısal hesaplamalar için
- `matplotlib`: Eğitim sürecindeki hata grafiğini çizmek için
- `sklearn`: Model performansını değerlendirmek için (Accuracy, Precision, Recall, F1-Score)

## Eğitim Süreci

Model, aşağıdaki OR kapısı girişleri ve çıkışları ile eğitilmektedir:

| Giriş A | Giriş B | Çıktı |
|---------|---------|-------|
|    0    |    0    |   0   |
|    0    |    1    |   1   |
|    1    |    0    |   1   |
|    1    |    1    |   1   |

### Modelin Temel Yapısı

- **Sigmoid Aktivasyon Fonksiyonu**: Ağırlıkları güncellerken ve tahmin yaparken kullanılan temel fonksiyon.
- **Eğitim**: Ağırlıklar, her bir örnek üzerinden iterasyon yaparak güncellenir. Eğitim sırasında toplam hata, her dönemde hesaplanır ve görselleştirilir.
- **Performans Ölçütleri**: Eğitim sonunda modelin doğruluğu (accuracy), hassasiyeti (precision), geri çağırma oranı (recall) ve F1 puanı hesaplanır.

## Kullanım

1. Gerekli kütüphanelerin kurulumunu yapın:
    ```bash
    pip install numpy matplotlib scikit-learn
    ```

2. Aşağıdaki kodu çalıştırarak modeli eğitin ve sonuçları görüntüleyin:
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics i
