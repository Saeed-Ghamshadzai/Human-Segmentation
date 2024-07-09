from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np

app = FastAPI()

@app.get('/health_checker')
async def health_checker():
    return JSONResponse(content={'status': 'running'})
