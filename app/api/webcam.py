import cv2
from utils.utils import InvTransform, apply_mask_overlay, segment_image, image_to_base64
from data_pytorch.preprocessor import PreprocessImage

processor = PreprocessImage()
inv_transform = InvTransform()

class WebcamCapture:
    def __init__(self, width=128, height=128, backend=cv2.CAP_DSHOW, model=None, device='cuda', preprocessor=PreprocessImage()):
        """
        Initializes the webcam capture object.
        :param width: Width of the capture frame.
        :param height: Height of the capture frame.
        :param backend: Backend used to open the webcam. Default is DirectShow.
        """
        self.model = model
        self.device = device
        self.width = width
        self.height = height
        self.backend = backend
        self.webcam = None
        self.is_started = False
        self.preprocessor = [preprocessor]

    def start(self):
        """Starts the webcam capture."""
        if not self.is_started:
            self.webcam = cv2.VideoCapture(0, self.backend)
            if not self.webcam.isOpened():
                raise Exception("Could not open webcam.")
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.is_started = True

    def stop(self):
        """Stops the webcam capture and releases resources."""
        if self.is_started:
            self.webcam.release()
            self.webcam = None
            self.is_started = False

    def get_frame(self):
        """Captures a frame from the webcam."""
        if not self.is_started:
            raise Exception("Webcam is not started. Call start() first.")
        
        ret, frame = self.webcam.read()
        if not ret:
            raise Exception("Failed to capture frame.")
        return frame

    def get_segmented_frame(self):
        """Returns the segmented version of the captured frame."""
        frame = self.get_frame()
        segmented_frame = self.segment_frame(frame)

        return segmented_frame

    def get_overlayed_frame(self):
        """Returns the overlayed version of the captured frame."""
        frame = self.get_frame()
        segmented_frame = self.segment_frame(frame)

        overlayed_frame = self.overlay_frame(frame, segmented_frame)
        return overlayed_frame

    def segment_frame(self, frame):
        """Segments the given frame. (Placeholder for actual segmentation logic)"""
        # Convert to grayscale as a placeholder for actual segmentation
        processed_frame = preprocessor(frame)
        segmented_frame = segment_image(self.model, processed_frame, self.device)
        return segmented_frame

    def overlay_frame(self, frame, segmented_frame):
        """Overlays the segmented frame onto the original frame."""
        # Ensure segmented frame is 3-channel to overlay
        processed_frame = preprocessor(frame).squeeze(0)
        inversed_transformation_image = inv_transform(processed_frame).permute(1, 2, 0).numpy()

        overlayed_frame = apply_mask_overlay(inversed_transformation_image, segmented_frame)
        return overlayed_frame

    def get_frames_base64(self):
        frame = self.get_frame()
        segmented_frame = self.get_segmented_frame()
        overlayed_frame = self.get_overlayed_frame()

        return (
            image_to_base64(frame),
            image_to_base64(segmented_frame),
            image_to_base64(overlayed_frame)
        )

# Example usage
if __name__ == "__main__":
    webcam = WebcamCapture()

    try:
        webcam.start()
        for _ in range(100):  # Capture 10 frames as a test
            frame = webcam.get_frame()
            segmented_frame = webcam.get_segmented_frame()
            overlayed_frame = webcam.get_overlayed_frame()

            # Display the frames (press 'q' to quit)
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Segmented Frame', segmented_frame)
            cv2.imshow('Overlayed Frame', overlayed_frame)

            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
    finally:
        webcam.stop()
        cv2.destroyAllWindows()
