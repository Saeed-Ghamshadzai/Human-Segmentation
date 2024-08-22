from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from moviepy.editor import VideoFileClip, AudioFileClip
from api.webcam import WebcamCapture
from api.segmentor.data import PreprocessImage
from api.segmentor.model import load_model, predict
from api.segmentor.utils import InvTransform, apply_mask_overlay
import requests
import torch
import cv2
import re
import os
import tqdm
import base64
from io import BytesIO
import numpy as np
import urllib.parse

app = FastAPI()

camera = WebcamCapture()

processor = PreprocessImage()
inv_transform = InvTransform()

# Path to the saved model weights
weights_path = "app\\api\\segmentor\\DeepLabV3-Model-V1.1.pth.tar"

# Initialize the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(weights_path, device)

user_upload_dir = os.path.join('app', 'api', 'uploads_dir')
os.makedirs(user_upload_dir, exist_ok=True)

test_images = ["C:\\Users\ZBook Studio\Pictures\\Jowhareh_galleries_3_poster_11e0fb75-eda7-4eb5-ab64-2636e5493170.jpeg",
            "C:\\Users\ZBook Studio\Pictures\\Jowhareh_galleries_3_poster_785fa3f4-e35d-429c-b021-89efe144fb38.jpeg"]

def image_to_base64(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def segment_image(image_array):
    mask = predict(model, image_array, device) * 255

    return mask
    
def segment_video(video_path):
    # Extract video directory, name, and extension
    video_dir, video_name = os.path.split(video_path)
    video_base, video_ext = os.path.splitext(video_name)
    
    # Load the video
    video_clip = VideoFileClip(video_path)
    
    # Get video properties
    fps = video_clip.fps
    width, height = video_clip.size
    audio_clip = video_clip.audio

    # Define the output video path
    segmented_video_path = os.path.join(video_dir, f"{video_base}_segmented{video_ext}")

    # Process each frame to grayscale
    def process_frame(frame):
        processed_frame = processor(frame)
        segmented_image = segment_image(processed_frame)

        original_image = inv_transform(processed_frame.squeeze(0)).permute(1, 2, 0).numpy()
        overlayed_image = apply_mask_overlay(original_image, segmented_image, format='RGB')

        return overlayed_image
    
    # Apply the frame processing
    processed_clip = video_clip.fl_image(process_frame)

    # Write the processed video with the original audio
    processed_clip = processed_clip.set_audio(audio_clip)
    processed_clip.write_videofile(segmented_video_path, codec='libx264', fps=fps)

    print(f"Processed video saved at: {segmented_video_path}")

    return segmented_video_path

# Health checker endpoint
@app.get('/health_checker')
async def health_checker():
    return JSONResponse(content={'status': 'running'})

# Test file endpoint
@app.get("/test-images/", response_class=HTMLResponse)
async def list_test_images():
    uploaded_files = [os.path.join(user_upload_dir, x) for x in os.listdir(user_upload_dir) if not x.endswith('segmented.mp4')]
    uploaded_file_links = "".join([f"""
    <li>
    <a href="/{'video-player' if uploaded_file.endswith('.mp4') else 'view-image'}/{urllib.parse.quote_plus(uploaded_file)}">{uploaded_file}</a>
    </li>
    """ for uploaded_file in uploaded_files])

    test_image_links = "".join([f"""
    <li>
    <a href="/view-image/{urllib.parse.quote_plus(image_name)}">{image_name}</a>
    </li>
    """ for image_name in test_images])

    return HTMLResponse(content=f"""
    <html>
    <body>
    <h1>Test Images</h1><ul>{test_image_links}</ul>
    <h1>Uploaded Files</h1><ul>{uploaded_file_links}</ul>
    </body>
    </html>
    """)

# Upload Form Endpoint
@app.get("/uploadfile/", response_class=HTMLResponse)
async def upload_file_form():
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <body>
    <h2>Upload File (Image / Video)</h2>
    <h3>Be careful with the size of the videos you upload! (try small videos)</h3>
    <form action="/uploadfile/" method="post" enctype="multipart/form-data">
        <label for="file">Choose a file:</label><br><br>
        <input type="file" name="file"><br><br>
        <label for="url">Or enter a URL:</label><br><br>
        <input type="text" name="url" placeholder="http://example.com/file.jpg"><br><br>
        <button type="submit">Upload</button>
    </form>
    </body>
    </html>
    """)

# File/URL Upload Handling Endpoint
@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = Form(None), url: str = Form(None)):
    try:
        # Extract file name using regex to match up to .jpeg, .jpg, .png
        regex = r".*?\.(jpg|jpeg|jfif|png|mp4|avi|mov|mkv)"
        
        if file.filename:
            match = re.search(regex, file.filename)
            if not match:
                raise HTTPException(status_code=400, detail="Input does not contain a valid image file extension")
            
            # Save uploaded file
            file_path = os.path.join(user_upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

        elif url:
            match = re.search(regex, url)
            if not match:
                raise HTTPException(status_code=400, detail="URL does not contain a valid image file extension")
            # Extract the matched portion of the URL
            image_url = match.group(0)

            # Download and save the file from URL
            response = requests.get(image_url)
            response.raise_for_status()  # Check for errors
            file_name = os.path.basename(image_url)
            file_path = os.path.join(user_upload_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")

        # Determine file type and create the appropriate response
        if file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return HTMLResponse(content=f"""
            <html>
            <body>
            <h1>Upload Successful</h1>
            <p>Uploaded video: {file_path}</p>
            <p><a href="/video-player/{urllib.parse.quote_plus(file_path)}">View Segmented Video</a></p>
            </body>
            </html>
            """)
        elif file_path.endswith(('.jpg', '.jpeg', '.png', '.jfif')):
            return HTMLResponse(content=f"""
            <html>
            <body>
            <h1>Upload Successful</h1>
            <p>Uploaded image: {file_path}</p>
            <p><a href="/view-image/{urllib.parse.quote_plus(file_path)}">View Segmented Image</a></p>
            </body>
            </html>
            """)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")
        
# View selected image endpoint
@app.get("/view-image/{image_name}", response_class=HTMLResponse)
async def view_image(image_name: str):
    decoded_image_name = urllib.parse.unquote_plus(image_name)
    image = cv2.imread(decoded_image_name)

    image = processor(image)
    segmented_image = segment_image(image)

    original_image = inv_transform(image).squeeze(0).permute(1, 2, 0).numpy()
    overlayed_image = apply_mask_overlay(original_image, segmented_image)

    original_image_base64 = image_to_base64(original_image)
    segmented_image_base64 = image_to_base64(segmented_image)
    overlayed_image_base64 = image_to_base64(overlayed_image)

    html_content = f"""
    <html>
    <body>
    <h1>Original and Segmented Image</h1>

    <div style="display: flex; gap: 50px;">
        
        <div>
            <h2>Original Image</h2>
            <img src="data:image/jpeg;base64,{original_image_base64}" alt="Original Image">
        </div>

        <div>
            <h2>Overlayed Image</h2>
            <img src="data:image/jpeg;base64,{overlayed_image_base64}" alt="Segmented Image">
        </div>

        <div>
            <h2>Segmented Image</h2>
            <img src="data:image/jpeg;base64,{segmented_image_base64}" alt="Segmented Image">
        </div>
    
    </div>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# View selected video endpoint
@app.get("/view-video/{video_name}")
async def view_video(video_name: str):
    video_path = urllib.parse.unquote_plus(video_name)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    # Open and stream the video file
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.get("/video-player/{video_name}", response_class=HTMLResponse)
async def video_player(video_name: str):
    segmented_video_name = segment_video(video_name)
    
    encoded_video_name = urllib.parse.quote_plus(video_name)
    encoded_segmented_video_name = urllib.parse.quote_plus(segmented_video_name)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body>
    <h1>Video Players</h1>
        
    <div style="display: flex; gap: 50px;">

        <div>
            <h2>Original Video</h2>
            <video width="640" height="480" controls>
            <source src="/view-video/{encoded_video_name}" type="video/mp4">
            Your browser does not support the video tag.
            </video>
        </div>

        <div>
            <h2>Segmented Video</h2>
            <video width="640" height="480" controls>
            <source src="/view-video/{encoded_segmented_video_name}" type="video/mp4">
            Your browser does not support the video tag.
            </video>
        </div>

    </div>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Webcam segmentation endpoint
@app.get("/start-webcam")
def start_webcam():
    """Starts the webcam."""
    camera.start()
    return {"status": "Webcam started"}

@app.get("/stop-webcam")
def stop_webcam():
    """Stops the webcam."""
    camera.stop()
    return {"status": "Webcam stopped"}

@app.get("/webcam-feed", response_class=HTMLResponse)
def webcam_feed():
    """Displays the original, segmented, and overlayed frames in a single row."""
    camera.start()

    # Capture the frames as base64 strings
    original_base64, segmented_base64, overlayed_base64 = camera.get_frames_base64()

    # Construct the HTML page
    html_content = f"""
    <html>
    <body>
    <h1>Webcam Feed</h1>
    <div style="display: flex; justify-content: space-between;">
        <div style="margin-right: 10px;">
            <h2>Original Frame</h2>
            <img src="data:image/jpeg;base64,{image_to_base64(original_base64)}" alt="Original Frame">
        </div>
        <div style="margin-right: 10px;">
            <h2>Segmented Frame</h2>
            <img src="data:image/jpeg;base64,{image_to_base64(segmented_base64)}" alt="Segmented Frame">
        </div>
        <div style="margin-right: 10px;">
            <h2>Overlayed Frame</h2>
            <img src="data:image/jpeg;base64,{image_to_base64(overlayed_base64)}" alt="Overlayed Frame">
        </div>
    </div>
    <script>
        setTimeout(function() {{
            window.location.reload(1);
        }}, 200);  // Refresh the page every 200 milliseconds to update frames
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.on_event("shutdown")
def shutdown_event():
    """Ensure webcam is stopped when the application shuts down."""
    camera.stop()