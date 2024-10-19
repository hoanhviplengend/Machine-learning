from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import hand_sign_recognize as model
app = FastAPI()

class PathInput(BaseModel):
    path: Optional[str] = r"A:\myProject\projectML\Check\1.jpg"

def run_action(path: str):
    result = model.predict_new(None, path)
    return result

@app.post("/run")  # Đảm bảo sử dụng @app.post
async def run(input: PathInput):
    result = run_action(input.path)
    return {"result": result}




# uvicorn FastAPI:app --reload  ---> run
# http://127.0.0.1:8000/docs
#click Post -> Try it out -> add your path
#exacute
