
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scripts')))
import hand_sign_recognize as model
from datetime import datetime
from hand_sign_recognize.Scripts import Read_and_writeJson as raw
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

path_history = "../Data/history/history_FastAPI.json"
data_history = raw.read_json(path_history)

@app.post("/predict")  # Đảm bảo sử dụng @app.post
async def run(input: PathInput):
    predict = model.predict_new(None, input.path)
    now = datetime.now()
    history = creat_object_history("predict", input.path, predict, now.isoformat())
    data_history.append(history)
    raw.write_json(path_history, data_history)
    return {"result": predict}

@app.post("/run_model")
async def run2(input: Modelintput):
    now = datetime.now()
    result = "True" if input.script == "run" else "False"
    history = creat_object_history("predict", input.script, result, now.isoformat())
    data_history.append(history)
    raw.write_json(path_history,data_history)
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
