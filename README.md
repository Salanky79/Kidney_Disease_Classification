# Kidney Disease Classification (MLOps Pipeline)

Hệ thống phân loại ảnh CT chẩn đoán bệnh về thận (4 lớp) áp dụng quy trình MLOps hoàn chỉnh: Transfer Learning với ResNet50, quản lý quy trình dữ liệu bằng DVC, theo dõi chỉ số thực nghiệm qua MLflow/DagsHub và triển khai tự động CI/CD lên AWS EC2 qua GitHub Actions.

---

## 🛠️ Công nghệ & Chỉ số
* **Kiến trúc mô hình:** ResNet50 (Transfer Learning)
* **Chỉ số đánh giá:** Loss, Accuracy, Precision, Recall, F1-Score (Macro Average)
* **MLOps Stack:** DVC, MLflow, DagsHub, Docker, GitHub Actions, AWS (ECR & EC2)

---

## 🔄 Quy trình phát triển (Workflows)
1. Cập nhật `config/config.yaml`
2. Cập nhật `params.yaml`
3. Cập nhật `entity` (`src/cnnClassifier/entity/config_entity.py`)
4. Cập nhật Configuration Manager (`src/cnnClassifier/config/configuration.py`)
5. Cập nhật Components (`src/cnnClassifier/components/`)
6. Cập nhật Pipeline (`src/cnnClassifier/pipeline/`)
7. Cập nhật `main.py`
8. Cập nhật `dvc.yaml`
9. Chạy ứng dụng Web `app.py`

---

## 🚀 Hướng dẫn cài đặt & Chạy cục bộ

### Bước 1: Clone Repository
```bash
git clone [https://github.com/Salanky79/Kidney_Disease_Classification.git](https://github.com/Salanky79/Kidney_Disease_Classification.git)
cd Kidney_Disease_Classification
```

### Bước 2: Khởi tạo môi trường Conda
```bash
conda create -n kidney python=3.10 -y
conda activate kidney
```

### Bước 3: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Thiết lập biến môi trường DagsHub / MLflow
**Linux / macOS:**
```bash
export MLFLOW_TRACKING_URI=[https://dagshub.com/Salanky79/Kidney_Disease_Classification.mlflow](https://dagshub.com/Salanky79/Kidney_Disease_Classification.mlflow)
export MLFLOW_TRACKING_USERNAME=Salanky79 
export MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
```

**Windows (CMD / PowerShell):**
```cmd
set MLFLOW_TRACKING_URI=[https://dagshub.com/Salanky79/Kidney_Disease_Classification.mlflow](https://dagshub.com/Salanky79/Kidney_Disease_Classification.mlflow)
set MLFLOW_TRACKING_USERNAME=Salanky79
set MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
```

### Bước 5: Thực thi Pipeline & Chạy ứng dụng
Chạy toàn bộ Pipeline:
```bash
python main.py
```
Hoặc khởi chạy lại với DVC:
```bash
dvc repro
```
Mở giao diện Web Flask:
```bash
python app.py
```

---

## ☁️ Triển khai CI/CD lên AWS qua GitHub Actions

### 1. Cấu hình IAM & ECR trên AWS
* Tạo IAM User với 2 quyền:
  * `AmazonEC2ContainerRegistryFullAccess`
  * `AmazonEC2FullAccess`
* Tạo ECR Repository lưu trữ Docker Image (Ví dụ: `kidney-disease`).

### 2. Cấu hình máy chủ EC2 (Ubuntu)
Mở terminal trên instance EC2 và cài đặt Docker:
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 3. Đăng ký Self-hosted Runner trên GitHub
Vào Repository: `Settings` > `Actions` > `Runners` > `New self-hosted runner` -> Chọn OS `Linux` và chạy các câu lệnh hướng dẫn trên máy EC2.

### 4. Khai báo GitHub Secrets
Vào Repository: `Settings` > `Secrets and variables` > `Actions` và thêm các biến:
* `AWS_ACCESS_KEY_ID`: `<YOUR_AWS_ACCESS_KEY>`
* `AWS_SECRET_ACCESS_KEY`: `<YOUR_AWS_SECRET_KEY>`
* `AWS_REGION`: `us-east-1`
* `AWS_ECR_LOGIN_URI`: `<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com`
* `ECR_REPOSITORY_NAME`: `kidney-disease`