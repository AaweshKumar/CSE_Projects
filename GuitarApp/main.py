import os
import random
from PIL import Image, ImageTk, ImageDraw, ImageFont
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

BASE_FOLDER = "Chords"

running = False
timer_id = None
delay = 3000
current_index = 0
current_level = "Beginner"
image_files = []

difficulty_map = {
  "A.png": "Beginner",
  "A7.png": "Beginner",
  "Am.png": "Beginner",
  "Asus2.png": "Beginner",
  "Asus4.png": "Beginner",
  "A5.png": "Intermediate",
  "A6.png": "Intermediate",
  "Am6.png": "Intermediate",
  "Am7.png": "Intermediate",
  "Amaj7.png": "Intermediate",
  "A9.png": "Advanced",
  "Aaug.png": "Advanced",
  "Adim.png": "Advanced",
  "Adim7.png": "Advanced",
  "Am7b5.png": "Advanced",

  "B.png": "Intermediate",
  "B7.png": "Intermediate",
  "Bm.png": "Intermediate",
  "Bm7.png": "Intermediate",
  "Bmaj7.png": "Intermediate",
  "Baug.png": "Advanced",
  "Bdim.png": "Advanced",
  "Bdim7.png": "Advanced",
  "Bm7b5.png": "Advanced",

  "C.png": "Beginner",
  "C7.png": "Beginner",
  "Csus4.png": "Beginner",
  "Cadd9.png": "Beginner",
  "Cm.png": "Intermediate",
  "C#m.png": "Intermediate",
  "C5.png": "Intermediate",
  "C6.png": "Intermediate",
  "Cm7.png": "Intermediate",
  "Cmaj7.png": "Intermediate",
  "C9.png": "Advanced",
  "Caug.png": "Advanced",
  "Cdim.png": "Advanced",
  "Cdim7.png": "Advanced",
  "Cm7b5.png": "Advanced",

  "D.png": "Beginner",
  "Dm.png": "Beginner",
  "D7.png": "Beginner",
  "Dsus2.png": "Beginner",
  "Dsus4.png": "Beginner",
  "Dadd9.png": "Beginner",
  "D5.png": "Intermediate",
  "D6.png": "Intermediate",
  "Dm6.png": "Intermediate",
  "Dm7.png": "Intermediate",
  "Dmaj7.png": "Intermediate",
  "D9.png": "Advanced",
  "Daug.png": "Advanced",
  "Ddim.png": "Advanced",
  "Ddim7.png": "Advanced",
  "Dm7b5.png": "Advanced",

  "E.png": "Beginner",
  "Em.png": "Beginner",
  "E7.png": "Beginner",
  "Esus2.png": "Beginner",
  "Esus4.png": "Beginner",
  "E5.png": "Intermediate",
  "E6.png": "Intermediate",
  "Em6.png": "Intermediate",
  "Em7.png": "Intermediate",
  "Ema7.png": "Intermediate",
  "E9.png": "Advanced",
  "Eaug.png": "Advanced",
  "Edim.png": "Advanced",
  "Edim7.png": "Advanced",
  "Em7b5.png": "Advanced",

  "F.png": "Intermediate",
  "Fm.png": "Intermediate",
  "F#m.png": "Intermediate",
  "F7.png": "Intermediate",
  "Fsus4.png": "Intermediate",
  "Fm7.png": "Intermediate",
  "Fmaj7.png": "Intermediate",
  "Faug.png": "Advanced",
  "Fdim.png": "Advanced",
  "Fdim7.png": "Advanced",
  "Fm7b5.png": "Advanced",

  "G.png": "Beginner",
  "G7.png": "Beginner",
  "Gsus2.png": "Beginner",
  "Gsus4.png": "Beginner",
  "Gadd9.png": "Beginner",
  "Gm.png": "Intermediate",
  "G5.png": "Intermediate",
  "G6.png": "Intermediate",
  "Gm7.png": "Intermediate",
  "Gmaj7.png": "Intermediate",
  "G9.png": "Advanced",
  "Gaug.png": "Advanced",
  "Gdim.png": "Advanced",
  "Gdim7.png": "Advanced",
  "Gm7b5.png": "Advanced"
}


def get_files(level):
    files = []
    if not os.path.exists(BASE_FOLDER):
        print("Folder not found:", BASE_FOLDER)
        return files
    for note in os.listdir(BASE_FOLDER):
        note_path = os.path.join(BASE_FOLDER, note)
        if not os.path.isdir(note_path):
            continue
        for f in os.listdir(note_path):
            if f.lower().endswith(".png") and f in difficulty_map and difficulty_map[f] == level:
                files.append(os.path.join(note_path, f))
    return files

def show_image(path):
    try:
        img = Image.open(path)
        img.thumbnail((1200, 800), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img, text="")
        image_label.image = tk_img
    except Exception as e:
        print(f"Can't load '{path}': {e}")

def stop():
    global running, timer_id
    running = False
    if timer_id:
        try:
            root.after_cancel(timer_id)
        except:
            pass
        timer_id = None

def set_delay():
    global delay
    try:
        delay = int(float(timer_entry.get()) * 1000)
    except ValueError:
        delay = 500

def random_image():
    global timer_id
    if not running or not image_files or not root.winfo_exists():
        stop()
        return
    path = random.choice(image_files)
    show_image(path)
    timer_id = root.after(delay, random_image)

def sequential_image():
    global timer_id, current_index
    if not running or not image_files or not root.winfo_exists():
        stop()
        return
    if current_index >= len(image_files):
        current_index = 0
    show_image(image_files[current_index])
    current_index += 1
    timer_id = root.after(delay, sequential_image)

def start_random():
    global running, image_files
    stop()
    image_files = get_files(current_level)
    if not image_files:
        image_label.config(text="No images found for this level", image="")
        return
    set_delay()
    running = True
    random_image()

def start_sequential():
    global running, image_files, current_index
    stop()
    image_files = get_files(current_level)
    if not image_files:
        image_label.config(text="No images found for this level", image="")
        return
    set_delay()
    running = True
    current_index = 0
    sequential_image()

def on_level_change(event=None):
    global current_level, image_files
    stop()
    current_level = difficulty.get()
    image_files = get_files(current_level)

def gradient_title(text, font_path=None, font_size=36):
    width, height = 900, 70
    img = Image.new("RGBA", (width, height), (10,10,15,0))
    if font_path is None or not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, font_size)

    start_color = (0,255,255)
    end_color = (0,127,255)

    bbox = font.getbbox(text)
    text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x, y = (width-text_w)//2, (height-text_h)//2

    grad = Image.new("RGBA", (text_w, text_h))
    draw = ImageDraw.Draw(grad)
    for i in range(text_w):
        r = int(start_color[0]+(end_color[0]-start_color[0])*i/text_w)
        g = int(start_color[1]+(end_color[1]-start_color[1])*i/text_w)
        b = int(start_color[2]+(end_color[2]-start_color[2])*i/text_w)
        draw.line([(i,0),(i,text_h)], fill=(r,g,b))
    
    mask = Image.new("L", (text_w, text_h), 0)
    ImageDraw.Draw(mask).text((0,0), text, font=font, fill=255)
    grad.putalpha(mask)
    img.paste(grad, (x,y), grad)
    return ImageTk.PhotoImage(img)

#UI
root = ttkb.Window(themename="darkly")
root.title("Chord Trainer")
root.geometry("1250x800")

bg_img = Image.open("C:\\Users\\AAWESH KUMAR\\Pictures\\Screenshots\\Screenshot 2025-12-13 151851.png") #Paste folder path here for bg
bg_img = bg_img.resize((1250, 800), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_img)

bg_label = ttkb.Label(image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.image = bg_photo 

style = ttkb.Style(theme="darkly")
style.configure("TFrame", background="#0a0a0f")
style.configure("TLabel", background="#0a0a0f", font=("Segoe UI",12))

frame = ttkb.Frame(root, padding=2, bootstyle="transparent")
frame.place(relx=0, rely=0, relwidth=1, relheight=1)


title_img = gradient_title("Chord Switching Trainer", font_size=40)
title_label = ttkb.Label(image=title_img, background="#0a0a0f")
title_label.pack(pady=5)
title_label.image = title_img


image_label = ttkb.Label(frame, text="Select Difficulty and Timing!!", anchor="center")
image_label.pack(expand=True, fill="both", pady=20)


controls = ttkb.Frame(frame, bootstyle="transparent")
controls.pack(fill="x", pady=10)

ttkb.Label(controls, text="Difficulty:", foreground="#00ffff", background="#0a0a0f").grid(row=0, column=0, padx=5, pady=5, sticky="w")
difficulty = ttkb.StringVar()
difficulty_box = ttkb.Combobox(controls, textvariable=difficulty, values=["Beginner","Intermediate","Advanced"],
                               state="readonly", width=18, bootstyle=PRIMARY)
difficulty_box.grid(row=0,column=1,padx=5,pady=5)
difficulty_box.current(0)
difficulty_box.bind("<<ComboboxSelected>>", on_level_change)

ttkb.Label(controls, text="Interval (sec):", foreground="#00ffff", background="#0a0a0f").grid(row=0, column=2, padx=5, pady=5, sticky="w")
timer_entry = ttkb.Entry(controls, width=8, bootstyle=INFO)
timer_entry.insert(0,"3")
timer_entry.grid(row=0,column=3,padx=5,pady=5)

ttkb.Button(controls, text="Random", command=start_random, style="Glass.TButton").grid(row=1,column=0,pady=15)
ttkb.Button(controls, text="Sequential", command=start_sequential, style="Glass.TButton").grid(row=1,column=1,pady=15)
ttkb.Button(controls, text="Stop", command=stop, style="Glass.TButton").grid(row=1,column=2,pady=15)

root.mainloop()
