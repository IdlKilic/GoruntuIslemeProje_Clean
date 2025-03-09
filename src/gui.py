import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from predict import predict_image


def load_and_predict():
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not image_path:
        return

    model_path = "./model/handwriting_model.h5"

    try:
        image = Image.open(image_path)
        display_image = image.copy()
        display_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(display_image)

        if hasattr(load_and_predict, 'image_label') and load_and_predict.image_label:
            load_and_predict.image_label.destroy()

        load_and_predict.image_label = tk.Label(root, image=photo)
        load_and_predict.image_label.image = photo
        load_and_predict.image_label.pack(pady=10)

        predicted_text = predict_image(model_path, image_path)

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, predicted_text)

        save_to_docx(predicted_text)

    except Exception as e:
        messagebox.showerror("Hata", f"Tahmin sırasında bir hata oluştu: {e}")


def save_to_docx(text):
    try:
        from docx import Document

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "tahmin_sonucu.docx")

        doc = Document()
        doc.add_paragraph(text)
        doc.save(output_file)

        messagebox.showinfo("Başarılı", f"Tahmin tamamlandı ve '{output_file}' dosyasına kaydedildi.")
    except ImportError:
        messagebox.showwarning("Uyarı", "python-docx kütüphanesi yüklü değil. Word belgesi oluşturulamadı.")
    except Exception as e:
        messagebox.showerror("Hata", f"Belge kaydedilirken bir hata oluştu: {e}")


root = tk.Tk()
root.title("El Yazısı Tanıma")
root.geometry("400x400")

button = tk.Button(root, text="Görüntü Yükle ve Tahmin Et", command=load_and_predict, height=2)
button.pack(pady=20)

result_label = tk.Label(root, text="Tahmin Sonucu:")
result_label.pack(pady=(20, 5))

result_text = tk.Text(root, height=4, width=40)
result_text.pack(pady=5)

load_and_predict.image_label = None

info_label = tk.Label(root, text="Not: Tahmin edilen metin 'output/tahmin_sonucu.docx'\ndosyasına kaydedilecektir.")
info_label.pack(pady=10)

root.mainloop()
