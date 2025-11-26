# Campus Placement Prediction

Dự án Machine Learning dự đoán kết quả tuyển dụng sinh viên dựa trên dataset "Factors Affecting Campus Placement" từ Kaggle.

## 📊 Tổng quan dự án

Dự án này xây dựng pipeline machine learning hoàn chỉnh để dự đoán sinh viên có được tuyển dụng (`Placed`) hay không (`Not Placed`) trong các buổi tuyển dụng tại trường, dựa trên các yếu tố học tập và cá nhân.

### Tính năng chính:
- ✅ Pipeline tiền xử lý dữ liệu đầy đủ
- ✅ Nhiều mô hình ML với hyperparameter tuning
- ✅ Cross-validation để đánh giá robust
- ✅ Đánh giá chi tiết với nhiều metrics
- ✅ Visualization kết quả
- ✅ Feature importance analysis
- ✅ Lưu/tải mô hình
- ✅ Code structure rõ ràng, modular
- ✅ Notebook có documentation đầy đủ
- ✅ Kết quả reproducible với fixed random seed

## 📁 Cấu trúc dự án

```
Campus-Placement/
├── data/                                    # Thư mục dữ liệu
│   ├── README.md                            # Hướng dẫn tải dataset
│   └── Placement_Data_Full_Class.csv        # Dataset (cần tải về)
├── notebooks/                               # Jupyter notebooks
│   ├── 01_EDA.ipynb                        # Phân tích dữ liệu khám phá
│   └── 02_Modeling.ipynb                   # Training và đánh giá mô hình
├── src/                                     # Source code modules
│   ├── __init__.py                         # Package initialization
│   ├── data.py                             # Utilities tải dữ liệu
│   ├── preprocess.py                       # Tiền xử lý dữ liệu
│   ├── train.py                            # Training với CV và tuning
│   └── evaluate.py                         # Đánh giá mô hình
├── models/                                  # Thư mục lưu models
├── outputs/                                 # Thư mục lưu kết quả
├── run_models.py                           # Script chạy models và xuất plots
├── Campus_Placement_EDA_and_Models.ipynb   # Notebook tổng hợp EDA và Models
├── requirements.txt                        # Python dependencies
├── README.md                               # File này
├── LICENSE                                 # License file
└── .gitignore                              # Git ignore file
```

## 🚀 Hướng dẫn sử dụng

### Yêu cầu

- Python 3.8 trở lên
- pip package manager

### Cài đặt

1. Clone repository:
```bash
git clone https://github.com/dongmino042/Campus-Placement.git
cd Campus-Placement
```

2. Tạo virtual environment (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

3. Cài đặt các packages cần thiết:
```bash
pip install -r requirements.txt
```

### Tải Dataset

**Lưu ý**: Dataset không được bao gồm trong repository. Bạn cần tải về từ Kaggle.

**Cách 1: Tải thủ công**
- Truy cập [trang dataset trên Kaggle](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- Tải và giải nén file ZIP
- Đặt file `Placement_Data_Full_Class.csv` vào thư mục `data/`

**Cách 2: Sử dụng Kaggle API**
```bash
pip install kaggle
# Thiết lập Kaggle credentials (xem data/README.md)
kaggle datasets download -d benroshan/factors-affecting-campus-placement
unzip factors-affecting-campus-placement.zip -d data/
```

## 📊 Cách chạy

### Cách 1: Sử dụng script Python (Nhanh chóng)

Chạy script `run_models.py` để train models và tạo visualizations:

```bash
python run_models.py --data data/Placement_Data_Full_Class.csv --outdir outputs
```

Script này sẽ:
- Tiền xử lý dữ liệu
- Train các mô hình classification (dự đoán Placed/Not Placed)
- Train các mô hình regression (dự đoán mức lương cho sinh viên được tuyển)
- Lưu models và plots vào thư mục `outputs/`

### Cách 2: Sử dụng Jupyter Notebook (Khuyến nghị để khám phá)

1. **Notebook tổng hợp (EDA + Models):**
```bash
jupyter notebook Campus_Placement_EDA_and_Models.ipynb
```
Notebook này bao gồm:
- Exploratory Data Analysis (EDA) đầy đủ
- Training và đánh giá classification models
- Training và đánh giá regression models
- Visualizations và insights

2. **Notebooks riêng lẻ (trong thư mục notebooks/):**
```bash
# Phân tích dữ liệu
jupyter notebook notebooks/01_EDA.ipynb

# Training và đánh giá mô hình
jupyter notebook notebooks/02_Modeling.ipynb
```

### Cách 3: Sử dụng Python modules (trong src/)

```python
from data import load_data
from preprocess import prepare_train_test_split
from train import train_all_models, select_best_model, save_model
from evaluate import evaluate_model, print_evaluation_metrics

# Load và preprocess
df = load_data()
X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(df)

# Train models với CV và hyperparameter tuning
models = train_all_models(X_train, y_train, cv=5)

# Chọn và lưu best model
best_model_name, best_model, best_score = select_best_model(models)
save_model(best_model, 'best_model.pkl')

# Đánh giá
metrics = evaluate_model(best_model, X_test, y_test, best_model_name)
print_evaluation_metrics(metrics)
```

## 🤖 Các mô hình được implement

Dự án implement và so sánh các mô hình machine learning:

1. **Logistic Regression**
   - Hyperparameters: C, penalty, solver
   - Training nhanh, kết quả dễ interpret

2. **Random Forest**
   - Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
   - Robust với overfitting, xử lý tốt các mối quan hệ phi tuyến

3. **XGBoost** (nếu có cài đặt)
   - Hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma
   - State-of-the-art gradient boosting, performance xuất sắc

Tất cả models được train với:
- **5-fold cross-validation** để đánh giá robust
- **Grid search** để hyperparameter tuning
- **ROC AUC** là metric optimization chính

## 📈 Metrics đánh giá

Models được đánh giá bằng các metrics:

- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1 Score**: Trung bình điều hòa của precision và recall
- **ROC AUC**: Area under the ROC curve

Visualizations bổ sung:
- Confusion matrix
- ROC curve
- Feature importance (cho tree-based models)
- Model comparison charts

## 📊 Thông tin Dataset

**Nguồn**: [Kaggle - Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)

**Features**:
- Điểm số học tập (SSC, HSC, Degree, MBA percentages)
- Board of education
- Loại degree và specialization
- Kinh nghiệm làm việc
- Điểm employability test
- Giới tính

**Target**: 
- Classification: Placement Status (Placed/Not Placed)
- Regression: Salary (cho sinh viên được tuyển)

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Notebooks**: Jupyter
- **Model Persistence**: Joblib

## 📄 License

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.

## 👨‍💻 Tác giả

**dongmino042**

## 🙏 Acknowledgments

- Dataset được cung cấp bởi [Ben Roshan](https://www.kaggle.com/benroshan) trên Kaggle
- Lấy cảm hứng từ các thách thức tuyển dụng thực tế

## 📞 Liên hệ

Để đặt câu hỏi hoặc feedback, vui lòng mở issue trên GitHub.

---

**Lưu ý**: Dataset không được bao gồm trong repository này. Vui lòng tải về từ Kaggle theo hướng dẫn ở trên. Dự án này dành cho mục đích giáo dục.
