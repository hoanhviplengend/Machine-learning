
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import hand_sign_recognize as model
app = FastAPI()

class PathInput(BaseModel):
    path: Optional[str] = r"A:\myProject\projectML\Check\1.jpg"

class Modelintput(BaseModel):
    script: Optional[str] = ""

def creat_object_history(name, input, predict, time):
    return {"name":name,
            "input" :input,
            "result" : predict,
            "time" : time
    }
def read_json(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:  # Kiểm tra nếu file tồn tại và không trống
        with open(file_path, 'r') as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                return []
    return []

# Hàm ghi dữ liệu vào file JSON
def write_to_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def run_action(path: str):
    predict = model.predict_new(None, path)
    return predict

path_history = "history.json"
data_history = read_json(path_history)

@app.post("/predict")  # Đảm bảo sử dụng @app.post
async def run(input: PathInput):
    now = datetime.now()
    predict = run_action(input.path)
    history = creat_object_history("predict", input.path, predict, now.isoformat())
    data_history.append(history)
    write_to_json(path_history, data_history)
    return {"result": predict}

@app.post("/run_model")
async def run2(input: Modelintput):
    now = datetime.now()
    result = "True" if input.script == "run" else "False"
    history = creat_object_history("predict", input.script, result, now.isoformat())
    data_history.append(history)
    write_to_json(path_history,data_history)
    if input.script == "run":
        model.run()
        return {"result": True}
    return {"result": False}

if __name__ == "__main__":
    print("go to docs: http://127.0.0.1:8000/docs")
    print("predict: http://127.0.0.1:8000/docs#/default/run_predict_post")
    print("run_model: http://127.0.0.1:8000/docs#/default/run2_run_model_post")
    uvicorn.run(app, host="127.0.0.1", port=8000)


# uvicorn FastAPI:app --reload  ---> run
# http://127.0.0.1:8000/docs
#click Post -> Try it out -> add your path
#exacute
