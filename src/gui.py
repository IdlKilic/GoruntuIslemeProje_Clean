import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from predict import predict_image


def load_and_predict():
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not image_path:
        return

    # Model yolunu belirle - proje dizinine göre ayarlanmış
    model_path = "./model/handwriting_model.h5"

    try:
        # Görüntüyü yükle ve göster
        image = Image.open(image_path)
        # Ekrana sığdırmak için yeniden boyutlandır (görüntü ön izleme)
        display_image = image.copy()
        display_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(display_image)

        # Eski görüntüyü temizle
        if hasattr(load_and_predict, 'image_label') and load_and_predict.image_label:
            load_and_predict.image_label.destroy()

        # Yeni görüntüyü göster
        load_and_predict.image_label = tk.Label(root, image=photo)
        load_and_predict.image_label.image = photo  # Referansı tutmak için
        load_and_predict.image_label.pack(pady=10)

        # Tahmini yap
        predicted_text = predict_image(model_path, image_path)

        # Sonuç alanını güncelle
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, predicted_text)

        # Word belgesine kaydet
        save_to_docx(predicted_text)

    except Exception as e:
        messagebox.showerror("Hata", f"Tahmin sırasında bir hata oluştu: {e}")


def save_to_docx(text):
    try:
        # Docx kütüphanesini import et
        from docx import Document

        # Output dizinini oluştur (eğer yoksa)
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Dosya yolunu oluştur
        output_file = os.path.join(output_dir, "tahmin_sonucu.docx")

        # Belgeyi oluştur ve kaydet
        doc = Document()
        doc.add_paragraph(text)
        doc.save(output_file)

        # Başarılı mesajı göster
        messagebox.showinfo("Başarılı", f"Tahmin tamamlandı ve '{output_file}' dosyasına kaydedildi.")
    except ImportError:
        messagebox.showwarning("Uyarı", "python-docx kütüphanesi yüklü değil. Word belgesi oluşturulamadı.")
    except Exception as e:
        messagebox.showerror("Hata", f"Belge kaydedilirken bir hata oluştu: {e}")


# Ana pencereyi oluştur
root = tk.Tk()
root.title("El Yazısı Tanıma")
root.geometry("400x500")

# Görüntü yükleme butonu
button = tk.Button(root, text="Görüntü Yükle ve Tahmin Et", command=load_and_predict, height=2)
button.pack(pady=20)

# Sonuç etiketi
result_label = tk.Label(root, text="Tahmin Sonucu:")
result_label.pack(pady=(20, 5))

# Sonuç metin alanı
result_text = tk.Text(root, height=4, width=40)
result_text.pack(pady=5)

# Görüntü gösterimi için initial değer
load_and_predict.image_label = None

# Bilgi notu
info_label = tk.Label(root, text="Not: Tahmin edilen metin 'output/tahmin_sonucu.docx'\ndosyasına kaydedilecektir.")
info_label.pack(pady=10)

root.mainloop()