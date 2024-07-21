from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from moviepy.editor import VideoFileClip, AudioFileClip
from app.api.webcam import VideoCamera
import cv2
import os
import base64
import numpy as np
import urllib.parse

app = FastAPI()

camera = VideoCamera()

user_upload_dir = os.path.join('app', 'api', 'uploads_dir')
os.makedirs(user_upload_dir, exist_ok=True)

test_images = ["C:\\Users\ZBook Studio\Pictures\\Jowhareh_galleries_3_poster_11e0fb75-eda7-4eb5-ab64-2636e5493170.jpeg",
            "C:\\Users\ZBook Studio\Pictures\\Jowhareh_galleries_3_poster_785fa3f4-e35d-429c-b021-89efe144fb38.jpeg"]

def image_to_base64(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def segment_image(image_array):
    segmented_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return segmented_image

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
        return frame
        # gray_frame = segment_image(frame)
        # return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
    
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

# Upload endpoint
@app.get("/uploadfile/", response_class=HTMLResponse)
async def upload_file_form():
    return HTMLResponse(content=f"""
    <!DOCTYPE html>

    <html>
    <body>
    <h2>Upload File (Image / Video)</h2>
    <h3>Be careful with the size of the videos you upload ! (try small videos)</h3>
    <form action="/uploadfile/" method="post" enctype="multipart/form-data">
    <input type="file" name="file">
    <button type="submit">Upload</button>
    </form>
    </body>
    </html>
    """)

@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile):
    file_path = os.path.join(user_upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Display HTML response with link to view uploaded file
    if file.filename.endswith('mp4'):
        return HTMLResponse(content=f"""
        <html>
        <body>
        <h1>Upload Successful</h1>
        <p>Uploaded video: {file_path}</p>
        <p><a href="/video-player/{urllib.parse.quote_plus(file_path)}">View Segmented Video</a></p>
        </body>
        </html>
        """)
    else:
        return HTMLResponse(content=f"""
        <html>
        <body>
        <h1>Upload Successful</h1>
        <p>Uploaded image: {file_path}</p>
        <p><a href="/view-image/{urllib.parse.quote_plus(file_path)}">View Segmented Image</a></p>
        </body>
        </html>
        """)

# View selected image endpoint
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
    
    <h2>Original Video</h2>
    <video width="640" height="480" controls>
      <source src="/view-video/{encoded_video_name}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    
    <h2>Segmented Video</h2>
    <video width="640" height="480" controls>
      <source src="/view-video/{encoded_segmented_video_name}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Webcam segmentation endpoint
def generate_webcam_feed():
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_segmented_webcam_feed():
    while True:
        frame = camera.get_segmented_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/webcam-feed/")
async def webcam_feed():
    return StreamingResponse(generate_webcam_feed(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/segmented-webcam-feed/")
async def segmented_webcam_feed():
    return StreamingResponse(generate_segmented_webcam_feed(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/webcam-player/", response_class=HTMLResponse)
async def webcam_player():
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
    <h1>Webcam Feeds</h1>
    
    <h2>Original Webcam Feed</h2>
    <img id="original" src="/webcam-feed/" style="width: 640px; height: 480px;">
    
    <h2>Segmented Webcam Feed</h2>
    <img id="segmented" src="/segmented-webcam-feed/" style="width: 640px; height: 480px;">
    
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.on_event("shutdown")
def shutdown_event():
    camera.stop()