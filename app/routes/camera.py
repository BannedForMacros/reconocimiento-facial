from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
import cv2
import asyncio

router = APIRouter(prefix="/api", tags=["camera"])

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 15)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

camera = VideoCamera()

def gen_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )