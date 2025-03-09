import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape, LSTM, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from PIL import Image
import json


def load_data(dataset_dir):
    labels_file = os.path.join(dataset_dir, "labels.txt")
    words_dir = os.path.join(dataset_dir, "words")

    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"'labels.txt' dosyası bulunamadı: {labels_file}")

    with open(labels_file, "r") as f:
        lines = f.readlines()

    images, labels = [], []

    # İngilizce karakterler ve boşluk
    all_characters = "abcdefghijklmnopqrstuvwxyz "
    char_to_index = {char: idx + 1 for idx, char in enumerate(all_characters)}

    # Karakter-indeks eşlemesini kaydet (tahmin için kullanılacak)
    char_map = {
        "characters": all_characters,
        "char_to_index": char_to_index,
        "index_to_char": {str(idx): char for char, idx in char_to_index.items()}
    }

    with open(os.path.join("./model", "char_map.json"), "w") as f:
        json.dump(char_map, f, indent=4)

    for line in lines:
        try:
            word_id, text = line.strip().split(" ", 1)
            image_path = os.path.join(words_dir, f"{word_id}.png")

            if os.path.exists(image_path):
                # Görüntüyü yükle ve ön işleme uygula
                image = Image.open(image_path).convert("L").resize((256, 64))

                # Metni küçük harflere dönüştür ve sadece tanımlı karakterleri kullan
                text = text.lower()
                label = [char_to_index[char] for char in text if char in char_to_index]

                if label:  # Etiket boş değilse ekle
                    images.append(np.array(image) / 255.0)  # Normalize et
                    labels.append(label)
            else:
                print(f"Görüntü dosyası bulunamadı: {image_path}")
        except Exception as e:
            print(f"Hatalı satır atlandı: {line.strip()} - Hata: {e}")

    print(f"Yüklenen görüntü sayısı: {len(images)}, Etiket sayısı: {len(labels)}")

    # İstatistikler
    label_lengths = [len(label) for label in labels]
    print(f"Ortalama kelime uzunluğu: {np.mean(label_lengths):.2f} karakter")
    print(f"En uzun kelime uzunluğu: {max(label_lengths)} karakter")

    # Etiketlere padding uygula
    max_length = max(label_lengths)
    labels_padded = pad_sequences(labels, maxlen=max_length, padding='post', value=0)

    # Maksimum kelime uzunluğunu kaydet (tahmin için kullanılacak)
    model_config = {
        "max_length": int(max_length),
        "vocab_size": len(char_to_index) + 1
    }

    with open(os.path.join("./model", "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    return np.array(images), np.array(labels_padded), max_length, len(char_to_index) + 1


def create_model(input_shape, max_length, vocab_size):
    # Kelime tanıma için geliştirilmiş model
    model = Sequential([
        # CNN katmanları
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Özellik çıkarımı tamamlandı
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),  # Aşırı öğrenmeyi önlemek için
        Dense(max_length * vocab_size, activation="softmax"),
        Reshape((max_length, vocab_size))
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    dataset_dir = "../dataset"

    # Model dizinini oluştur
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    # Veri setini yükle
    images, labels, max_length, vocab_size = load_data(dataset_dir)

    # Veri şekillerini kontrol et ve göster
    images = np.expand_dims(images, axis=-1)  # Kanal boyutunu ekle
    print(f"Görüntü boyutu: {images.shape}")
    print(f"Etiket boyutu: {labels.shape}")
    print(f"Maksimum kelime uzunluğu: {max_length}")
    print(f"Kelime dağarcığı boyutu: {vocab_size}")

    # Veri setini eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Modeli oluştur
    model = create_model(input_shape=(64, 256, 1), max_length=max_length, vocab_size=vocab_size)

    # Model mimarisini göster
    model.summary()

    # Modeli eğit
    history = model.fit(
        X_train, y_train,
        epochs=15,  # Epoch sayısını arttırabilirsiniz
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Modeli kaydet
    model_path = os.path.join(model_dir, "handwriting_model.h5")
    model.save(model_path)
    print(f"Model başarıyla kaydedildi: {model_path}")

    # Test örnekleri üzerinde doğruluk kontrolü yap
    print("\nTest örnekleri üzerinde değerlendirme:")
    evaluation = model.evaluate(X_test, y_test)
    print(f"Test Kaybı: {evaluation[0]:.4f}")
    print(f"Test Doğruluğu: {evaluation[1]:.4f}")


if __name__ == "__main__":
    main()