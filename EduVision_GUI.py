import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from EduVision import main_video_processing
import time

class RoundButton(ttk.Button):
    def __init__(self, master=None, **kw):
        ttk.Button.__init__(self, master, **kw)
        self.config(style='Round.TButton')

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EduVision")
        self.root.geometry("800x600")  # Set the initial window size

        # Set the theme
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')

        # Generate the background image
        self.background_image = Image.new('RGBA', (800, 600), color='#6b97f7')
        self.draw = ImageDraw.Draw(self.background_image)

        # Create a canvas and add the background image
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Create the gradient animation
        self.gradient_animation()

        self.label = tk.Label(self.root, text="Choose a video file to analyze:", fg='black')
        self.label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        # Create a custom style for the button with round edges
        self.style.configure('Round.TButton', background='#4d4d4d', foreground='white', font=('Helvetica', 12), borderwidth=0, focuscolor='#4d4d4d', bordercolor='#4d4d4d', relief='flat', padding=5)
        self.style.map('Round.TButton', background=[('active', '!disabled', '#2E86C1')])

        self.browse_button = RoundButton(self.root, text="Upload", command=self.browse_file)
        self.browse_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.result_label = tk.Label(self.root, text="", bg='#2E86C1', fg='white')
        self.result_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            self.analyze_video(file_path)

    def analyze_video(self, video_path):
        self.result_label.config(text="Analyzing...", fg='white')
        self.root.update_idletasks()
        try:
            main_video_processing(video_path)
            self.result_label.config(text="Analysis complete!", fg='#90EE90')
        except Exception as e:
            self.result_label.config(text=f"Error: {e}", fg='red')

    def gradient_animation(self):
        colors = ["#6b97f7", "#7525e2", "#f7137e"]
        self.draw.rectangle([(0, 0), (800, 600)], fill=colors[0])
        self.background_photo = ImageTk.PhotoImage(self.background_image.filter(ImageFilter.BLUR))  # Apply blur
        self.canvas.create_image(0, 0, image=self.background_photo, anchor=tk.NW)

        color_index = 0
        step = 0
        max_steps = 180

        def update_gradient():
            nonlocal color_index, step

            start_color = tuple(int(colors[color_index][i:i+2], 16) for i in (1, 3, 5))
            end_color = tuple(int(colors[(color_index + 1) % len(colors)][i:i+2], 16) for i in (1, 3, 5))

            red = int(start_color[0] * (1 - step / max_steps) + end_color[0] * (step / max_steps))
            green = int(start_color[1] * (1 - step / max_steps) + end_color[1] * (step / max_steps))
            blue = int(start_color[2] * (1 - step / max_steps) + end_color[2] * (step / max_steps))

            gradient_color = (red, green, blue)
            self.draw.rectangle([(0, 0), (800, 600)], fill=gradient_color)
            self.background_photo = ImageTk.PhotoImage(self.background_image.filter(ImageFilter.BLUR))  # Apply blur
            self.canvas.create_image(0, 0, image=self.background_photo, anchor=tk.NW)

            step += 1
            if step > max_steps:
                step = 0
                color_index = (color_index + 1) % len(colors)

            self.root.after(30, update_gradient)

        update_gradient()

root = tk.Tk()
app = VideoAnalyzerApp(root)
root.mainloop()
