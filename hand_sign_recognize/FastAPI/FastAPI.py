

import urllib.request
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scripts')))
import predict as model
from datetime import datetime
from hand_sign_recognize.Scripts import Read_and_writeJson as raw
app = FastAPI()
path_history = "../Data/history/history_FastAPI.json"
model_path = "../Scripts/best_model_lenet.h5"
data_history = raw.read_json(path_history)

class PathInput(BaseModel):
    path: Optional[str] = r"A:\myProject\projectML\Check\1.jpg"

class Modelintput(BaseModel):
    script: Optional[str] = ""

def creat_object_history(name, input, predict,completed, time):
    return {"name":name,
            "input" :input,
            "predict":predict,
            "completed" :completed,
            "time" : time
    }

@app.post("/predict")  # Đảm bảo sử dụng @app.post
async def run(input: PathInput):
    raw_input = input.path
    try:
        urllib.request.urlretrieve(input.path, 'test.jpg')
        input.path = "test.jpg"
    except Exception as e:
        print(e)
    completed, predict = model.predict_by_photo(model_path, input.path)
    now = datetime.now()
    history = creat_object_history("predict image from file_path",
                                   raw_input,
                                   predict,
                                   completed,
                                   now.isoformat())
    data_history.append(history)
    raw.write_json(path_history, data_history)
    return history



@app.post("/run_model")
async def run2(input: Modelintput):
    now = datetime.now()
    if input.script == "video" or input.script == "photo":
        completed, preddict = model.predict_by_video(model_path,input.script)
        history = creat_object_history("predict from camera",
                                       input.script,
                                       preddict,
                                       completed,
                                       now.isoformat())
        data_history.append(history)
        raw.write_json(path_history, data_history)
        return history
    return creat_object_history("predict from camera",
                                       input.script,
                                       ["None"],
                                       False,
                                       now.isoformat())

if __name__ == "__main__":
    print("go to docs: http://127.0.0.1:8000/docs")
    print("predict: http://127.0.0.1:8000/docs#/default/run_predict_post")
    print("run_model: http://127.0.0.1:8000/docs#/default/run2_run_model_post")
    uvicorn.run(app, host="127.0.0.1", port=8000)


# uvicorn FastAPI:app --reload  ---> run
# http://127.0.0.1:8000/docs
#click Post -> Try it out -> add your path
#exacute
