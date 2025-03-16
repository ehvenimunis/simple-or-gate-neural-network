import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_or_gate(epochs, alpha):
    # OR Kapısı Eğitim Verisi
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    
    # Ağırlıkları rastgele başlatma
    np.random.seed(42)
    w = np.random.uniform(-1, 1, (2, 1))
    b = np.random.uniform(-1, 1, (1,))
    
    error_history = []
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            # İleri besleme
            net = np.dot(X[i], w) + b
            output = sigmoid(net)
            
            # Hata hesaplama
            error = y[i] - output
            total_error += error**2
            
            # Ağırlık güncelleme
            delta = error * sigmoid_derivative(output)
            w += alpha * delta * X[i].reshape(-1, 1)
            b += alpha * delta
        
        error_history.append(total_error.sum())
        if epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch}, Hata: {total_error.sum()}")
    
    # Hata grafiği
    plt.plot(range(epochs), error_history, label=f'Alpha: {alpha}')
    plt.xlabel("Epochs")
    plt.ylabel("Toplam Hata")
    plt.legend()
    
    # Modelin tahminlerini görüntüleme ve değerlendirme
    predictions = sigmoid(np.dot(X, w) + b)
    rounded_predictions = np.round(predictions)
    
    accuracy = accuracy_score(y, rounded_predictions)
    precision = precision_score(y, rounded_predictions)
    recall = recall_score(y, rounded_predictions)
    f1 = f1_score(y, rounded_predictions)
    
    print("Tahminler:")
    print(rounded_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return rounded_predictions

# Farklı parametreleri deneme
plt.figure(figsize=(10, 5))
train_or_gate(epochs=500, alpha=0.01)
train_or_gate(epochs=500, alpha=0.1)
train_or_gate(epochs=500, alpha=1)
train_or_gate(epochs=500, alpha=10)

plt.show()