import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from detector import HumanEmotionDetector


class VisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Human & Emotion Detection")
        self.geometry("960x720")
        self.resizable(False, False)

        self.detector = HumanEmotionDetector()
        self.capture = None
        self.image_path = None
        self.current_image = None
        self.video_job = None

        self._build_ui()

    def _build_ui(self):
        control_frame = tk.Frame(self, bg="#1e1e1e", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Vision Detection", fg="white", bg="#1e1e1e", font=("Segoe UI", 18, "bold")).pack(pady=(0, 12))

        buttons = [
            ("Start Camera", self.start_camera),
            ("Stop Camera", self.stop_camera),
            ("Upload Image", self.upload_image),
            ("Process Image", self.process_image),
        ]

        for label, action in buttons:
            button = tk.Button(control_frame, text=label, width=20, height=2, command=action, bg="#0078d4", fg="white", font=("Segoe UI", 11, "bold"))
            button.pack(pady=8)

        self.status_label = tk.Label(control_frame, text="Status: Ready", fg="white", bg="#1e1e1e", font=("Segoe UI", 11))
        self.status_label.pack(pady=(16, 0))

        preview_label = tk.Label(control_frame, text="Preview", fg="white", bg="#1e1e1e", font=("Segoe UI", 14, "bold"))
        preview_label.pack(pady=(24, 6))

        self.preview_canvas = tk.Canvas(control_frame, width=300, height=200, bg="#2d2d2d")
        self.preview_canvas.pack()

        display_frame = tk.Frame(self, bg="#2b2b2b")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(display_frame, width=640, height=640, bg="black")
        self.video_canvas.pack(padx=10, pady=10)

    def start_camera(self):
        if self.capture and self.capture.isOpened():
            return
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the webcam.")
            return
        self.status_label.config(text="Status: Camera running")
        self._update_camera_frame()

    def stop_camera(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None
        if self.video_job:
            self.after_cancel(self.video_job)
            self.video_job = None
        self.status_label.config(text="Status: Camera stopped")
        self.video_canvas.delete("all")

    def upload_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*")]
        path = filedialog.askopenfilename(title="Choose an image", filetypes=filetypes)
        if not path:
            return
        self.image_path = path
        self._display_preview(path)
        self.status_label.config(text=f"Status: Image loaded")

    def process_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        frame = cv2.imread(self.image_path)
        if frame is None:
            messagebox.showerror("File Error", "Could not open the selected image.")
            return
        annotated = self.detector.annotate_frame(frame)
        self._show_frame(annotated)
        self.status_label.config(text="Status: Image processed")

    def _update_camera_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        
        try:
            ret, frame = self.capture.read()
            if not ret:
                self.status_label.config(text="Status: Camera frame not available")
                self.video_job = self.after(20, self._update_camera_frame)
                return

            # Resize frame for faster processing (reduce to 480 width)
            h, w = frame.shape[:2]
            if w > 480:
                scale = 480 / w
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Annotate the frame
            annotated = self.detector.annotate_frame(frame)
            
            # Display the frame
            self._show_frame(annotated)
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)[:30]}")
        
        # Schedule next frame update with proper timing
        self.video_job = self.after(33, self._update_camera_frame)  # ~30 FPS

    def _show_frame(self, frame):
        """Display frame on canvas (fast method)"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create image - use fast conversion
            image = Image.fromarray(frame_rgb)
            
            # Use NEAREST for fast resizing (instead of LANCZOS)
            display_size = (640, int(640 * frame_rgb.shape[0] / frame_rgb.shape[1]))
            if display_size[1] > 640:
                display_size = (int(640 * frame_rgb.shape[1] / frame_rgb.shape[0]), 640)
            
            image = image.resize(display_size, Image.NEAREST)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(320, 320, image=self.current_image)
        except Exception as e:
            self.status_label.config(text=f"Status: Display error")

    def _display_preview(self, path):
        image = Image.open(path)
        image.thumbnail((300, 200), Image.ANTIALIAS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_canvas.create_image(150, 100, image=self.preview_image)

    def on_close(self):
        self.stop_camera()
        self.detector.shutdown()
        self.destroy()


if __name__ == "__main__":
    app = VisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
