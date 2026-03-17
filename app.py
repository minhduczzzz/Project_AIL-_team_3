import os
import threading
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import customtkinter as ctk
import wikipedia

# Import kiến trúc mô hình của bạn
from model import DogBreedResNet

# --- CẤU HÌNH GIAO DIỆN HIỆN ĐẠI ---
ctk.set_appearance_mode("Dark")  # Chế độ tối (Dark Mode)
ctk.set_default_color_theme("blue")  # Màu chủ đạo

class DogBreedApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Dog Breed Recognizer - Nhận Diện Giống Chó")
        self.geometry("800x600")
        self.resizable(False, False)

        # 1. Khởi tạo các biến chứa mô hình
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.transform = None

        self.setup_ui()
        
        # Tải mô hình ngay khi mở app
        self.load_model()

    def setup_ui(self):
        # Khung chính chia làm 2 phần: Trái (Ảnh) và Phải (Thông tin)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL TRÁI (Hiển thị ảnh & Nút chọn ảnh) ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=15)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.image_label = ctk.CTkLabel(self.left_frame, text="Chưa có ảnh nào được chọn", width=300, height=300, corner_radius=10, fg_color="gray20")
        self.image_label.pack(pady=40, padx=20)

        self.upload_btn = ctk.CTkButton(self.left_frame, text="Tải ảnh từ máy tính", font=("Arial", 16, "bold"), height=40, command=self.upload_image)
        self.upload_btn.pack(pady=10)

        # --- PANEL PHẢI (Hiển thị kết quả & Wikipedia) ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.right_frame, text="KẾT QUẢ PHÂN TÍCH", font=("Arial", 24, "bold"), text_color="#3498db")
        self.title_label.pack(pady=(20, 10))

        self.breed_label = ctk.CTkLabel(self.right_frame, text="Giống chó: ---", font=("Arial", 18))
        self.breed_label.pack(pady=5, anchor="w")

        self.confidence_label = ctk.CTkLabel(self.right_frame, text="Độ tự tin: ---", font=("Arial", 18))
        self.confidence_label.pack(pady=5, anchor="w")

        self.wiki_label = ctk.CTkLabel(self.right_frame, text="Thông tin Bách khoa toàn thư:", font=("Arial", 16, "bold"))
        self.wiki_label.pack(pady=(20, 5), anchor="w")

        self.wiki_textbox = ctk.CTkTextbox(self.right_frame, width=350, height=250, font=("Arial", 14), wrap="word")
        self.wiki_textbox.pack(pady=5)
        self.wiki_textbox.insert("0.0", "Hãy tải một bức ảnh lên để AI bắt đầu nhận diện và tra cứu thông tin...")
        self.wiki_textbox.configure(state="disabled") # Không cho người dùng gõ vào đây

    def load_model(self):
        """Tải mô hình vào bộ nhớ (Chỉ chạy 1 lần khi mở app)"""
        checkpoint_path = "training_models/best_resnet.pth"
        
        if not os.path.exists(checkpoint_path):
            self.wiki_textbox.configure(state="normal")
            self.wiki_textbox.delete("0.0", "end")
            self.wiki_textbox.insert("0.0", f"❌ LỖI: Không tìm thấy file '{checkpoint_path}'. Hãy kiểm tra lại.")
            self.wiki_textbox.configure(state="disabled")
            self.upload_btn.configure(state="disabled")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.class_to_idx = checkpoint["class_to_idx"]
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
            num_classes = len(self.class_to_idx)

            self.model = DogBreedResNet(num_classes=num_classes, pretrained=False).to(self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()

            self.transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Lỗi tải mô hình: {e}")

    def upload_image(self):
        """Mở cửa sổ chọn file ảnh"""
        file_path = ctk.filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            self.display_image(file_path)
            self.predict_image(file_path)

    def display_image(self, file_path):
        """Hiển thị ảnh lên giao diện"""
        img = Image.open(file_path)
        # Resize ảnh để hiển thị vừa vặn trong khung 300x300
        ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))
        self.image_label.configure(image=ctk_image, text="")

    def predict_image(self, file_path):
        """Chạy dự đoán AI"""
        self.breed_label.configure(text="Đang phân tích...")
        self.confidence_label.configure(text="Đang phân tích...")
        
        self.wiki_textbox.configure(state="normal")
        self.wiki_textbox.delete("0.0", "end")
        self.wiki_textbox.insert("0.0", "Đang phân tích ảnh và tra cứu dữ liệu...")
        self.wiki_textbox.configure(state="disabled")
        
        # Cập nhật giao diện ngay lập tức
        self.update_idletasks()

        try:
            image = Image.open(file_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs[0], dim=0)
                max_prob, top_idx = torch.max(probabilities, dim=0)
                
                predicted_breed = self.idx_to_class[top_idx.item()]
                confidence = max_prob.item() * 100

            # Viết hoa chữ cái đầu cho đẹp
            display_name = predicted_breed.replace("_", " ").title()
            
            self.breed_label.configure(text=f"Giống chó: {display_name} 🐶")
            self.confidence_label.configure(text=f"Độ tự tin: {confidence:.2f}% 📊")

            # Chạy hàm gọi Wikipedia ở một luồng (thread) riêng để không làm đơ giao diện
            threading.Thread(target=self.fetch_wikipedia_info, args=(predicted_breed,)).start()

        except Exception as e:
            self.breed_label.configure(text="Lỗi phân tích!")
            print(f"Lỗi dự đoán: {e}")

    def fetch_wikipedia_info(self, breed_name):
        """Tra cứu Wikipedia trong background"""
        search_term = breed_name.replace("_", " ").replace("-", " ") + " dog"
        info_text = ""
        
        try:
            wikipedia.set_lang("vi")
            info_text = wikipedia.summary(search_term, sentences=4)
        except wikipedia.exceptions.PageError:
            try:
                wikipedia.set_lang("en")
                info_text = wikipedia.summary(search_term, sentences=4)
                info_text += "\n\n*(Lưu ý: Wikipedia chưa có bài Tiếng Việt cho giống này, đang hiển thị bản Tiếng Anh)*"
            except:
                info_text = "Xin lỗi, hiện chưa có thông tin bách khoa toàn thư cho giống chó này."
        except wikipedia.exceptions.DisambiguationError:
            info_text = "Tên giống chó này quá chung chung, Wikipedia không thể xác định chính xác bài viết."
        except Exception:
            info_text = "Không có kết nối mạng hoặc Wikipedia đang lỗi."

        # Đẩy kết quả lên giao diện
        self.wiki_textbox.configure(state="normal")
        self.wiki_textbox.delete("0.0", "end")
        self.wiki_textbox.insert("0.0", info_text)
        self.wiki_textbox.configure(state="disabled")

if __name__ == "__main__":
    app = DogBreedApp()
    app.mainloop()