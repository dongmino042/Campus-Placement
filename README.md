# Campus-Placement

Dự án phân tích và dự đoán kết quả tuyển dụng (Placed / Not Placed) sử dụng bộ dữ liệu "Factors Affecting Campus Placement" từ Kaggle.

Nội dung thêm bởi PR này:
- `Campus_Placement_EDA_and_Models.ipynb` — Notebook đầy đủ (EDA, tiền xử lý, huấn luyện và đánh giá mô hình).
- `run_models.py` — Script Python để chạy huấn luyện (classification + regression) và xuất biểu đồ, lưu models.
- `requirements.txt` — Các package cần thiết.

Hướng dẫn nhanh:
1. Đặt file dataset `Placement_Data_Full_Class.csv` vào thư mục gốc của repo hoặc chỉnh đường dẫn trong script/notebook.
2. Tạo môi trường ảo và cài dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy script để tạo outputs:
   ```bash
   python run_models.py --data Placement_Data_Full_Class.csv --outdir outputs
   ```
4. Mở notebook `Campus_Placement_EDA_and_Models.ipynb` để xem EDA và code tương tác.

Ghi chú:
- PR này không thêm dataset (do dataset đã được cung cấp bởi người dùng). Các file models/outputs sẽ được sinh khi chạy script.
