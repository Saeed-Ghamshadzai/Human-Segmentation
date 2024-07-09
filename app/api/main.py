from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import os
import base64
import numpy as np
import urllib.parse

app = FastAPI()

@app.get('/health_checker')
async def health_checker():
    return JSONResponse(content={'status': 'running'})

test_images = ["C:\\Users\ZBook Studio\Pictures\\Jowhareh_galleries_3_poster_11e0fb75-eda7-4eb5-ab64-2636e5493170.jpeg",
            "C:\\Users\ZBook Studio\Pictures\\Jowhareh_galleries_3_poster_785fa3f4-e35d-429c-b021-89efe144fb38.jpeg"]

def image_to_base64(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def segment_image(image_array):
    segmented_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return segmented_image

@app.get("/test-images", response_class=HTMLResponse)
async def list_test_images():
    image_links = "".join([f'<li><a href="/view-image/{urllib.parse.quote_plus(image_name)}">{image_name}</a></li>' for image_name in test_images])
    html_content = f"<html><body><h1>Test Images</h1><ul>{image_links}</ul></body></html>"
    return HTMLResponse(content=html_content)

# HTML form for file upload
upload_form = """
<!DOCTYPE html>
<html>
<body>
<h2>Upload File</h2>
<form action="/uploadfile/" method="post" enctype="multipart/form-data">
  <input type="file" name="file">
  <button type="submit">Upload</button>
</form>
</body>
</html>
"""

# Upload endpoint
@app.get("/uploadfile/", response_class=HTMLResponse)
async def upload_file_form():
    return upload_form

@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile):
    user_upload_dir = os.path.join('app', 'api', 'uploads_dir')
    os.makedirs(user_upload_dir, exist_ok=True)

    file_path = os.path.join(user_upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Display HTML response with link to view uploaded image
    return HTMLResponse(content=f"""
    <html>
    <body>
    <h1>Upload Successful</h1>
    <p>Uploaded file: {file_path}</p>
    <p><a href="/view-image/{urllib.parse.quote_plus(file_path)}">View Uploaded Image</a></p>
    </body>
    </html>
    """)

@app.get("/view-image/{image_name}", response_class=HTMLResponse)
async def view_image(image_name: str):
    decoded_image_name = urllib.parse.unquote_plus(image_name)
    image = cv2.imread(decoded_image_name)
    segmented_image = segment_image(image)

    original_image_base64 = image_to_base64(image)
    segmented_image_base64 = image_to_base64(segmented_image)

    html_content = f"""
    <html>
    <body>
    <h1>Original and Segmented Image</h1>
    <h2>Original Image</h2>
    <img src="data:image/jpeg;base64,{original_image_base64}" alt="Original Image">
    <h2>Segmented Image</h2>
    <img src="data:image/jpeg;base64,{segmented_image_base64}" alt="Segmented Image">
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)