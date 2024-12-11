import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import uvicorn

# Thêm các đường dẫn module vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scripts')))

# Import các module cần thiết
from hand_sign_recognize.src import predict as model
from hand_sign_recognize.Scripts import Read_and_writeJson as raw

# Cấu hình ứng dụng FastAPI
app = FastAPI()

# Đường dẫn và thông tin cấu hình
path_history = "../Data/history/history_FastAPI.json"
model_path = "../Scripts/best_model_lenet.h5"
data_history = raw.read_json(path_history)

# Hàm tạo đối tượng lịch sử
def creat_object_history(name, input_data, predict, completed, time):
    return {
        "name": name,
        "input": input_data,
        "predict": predict,
        "completed": completed,
        "time": time
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Dự đoán từ một file ảnh được upload."""
    try:
        # Lưu file ảnh tạm thời
        local_image_path = f"temp_{file.filename}"
        with open(local_image_path, "wb") as f:
            f.write(await file.read())

        # Gọi model để dự đoán
        completed, predict = model.predict_by_photo(model_path, local_image_path)

        # Lưu lịch sử
        now = datetime.now()
        history = creat_object_history(
            "Predict image from local file",
            file.filename,
            predict,
            completed,
            now.isoformat()
        )
        data_history.append(history)
        raw.write_json(path_history, data_history)

        # Xóa file tạm
        os.remove(local_image_path)

        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/run_model")
async def run_model(request: Request):
    """Dự đoán từ camera/video."""
    input_model = request.query_params.get("input_model")
    if not input_model:
        raise HTTPException(status_code=400, detail="input_model is required.")

    now = datetime.now()
    try:
        if input_model in ["video", "photo"]:
            completed, predict = model.predict_by_video(model_path, input_model)
            history = creat_object_history(
                "Predict from camera",
                input_model,
                predict,
                completed,
                now.isoformat()
            )
            data_history.append(history)
            raw.write_json(path_history, data_history)
            return history
        else:
            raise HTTPException(status_code=400, detail="Invalid input_model value.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
if __name__ == "__main__":
    print("Go to docs: http://127.0.0.1:8000/docs")
    print("Predict: http://127.0.0.1:8000/docs#/default/predict_image_predict_post")
    print("Run model: http://127.0.0.1:8000/docs#/default/run_model_run_model_post")
    uvicorn.run(app, host="127.0.0.1", port=8000)
