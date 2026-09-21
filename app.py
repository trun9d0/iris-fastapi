from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Khởi tạo mô hình đã lưu từ Bước 1
model = joblib.load("svm_model.pkl")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

# Khai báo cấu trúc dữ liệu đầu vào
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Từ điển ánh xạ nhãn dự đoán
species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

# Tạo các endpoint cơ bản
@app.get("/")
def home():
    return {"message": "Iris SVM API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Endpoint dự đoán
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    prediction = int(model.predict(features)[0])
    return {
        "class_id": prediction,
        "prediction": species[prediction],
    }