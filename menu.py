import tkinter
from tkinter import*
import ui, registration
from PIL import Image, ImageTk


logo_dim = (int(504/3), int(160/3))
background = 'ghost white'

pady = (20, 10)
padx = (20, 20)

root = tkinter.Tk()
root.geometry("350x400")
root.title("Automotive face recognition")

root.configure(background='ghost white')

def recognition(stereo):
    ui.run(stereo)

def sign():
    registration.run(root)


vision_image = Image.open("img/vision.png")
idneo_image = Image.open("img/idneo.png")

vision_image = vision_image.resize(logo_dim, Image.ANTIALIAS)
idneo_image = idneo_image.resize(logo_dim, Image.ANTIALIAS)

vision_image = ImageTk.PhotoImage(vision_image)
idneo_image = ImageTk.PhotoImage(idneo_image)

vision = Label(root, image=vision_image, background='ghost white')
idneo = Label(root, image=idneo_image, background='ghost white')

button_2d = tkinter.Button(root, text="2D Face Recognition", command=lambda : recognition(False), height=2, width=35)
button_3d = tkinter.Button(root, text="3D Face Recognition", command=lambda : recognition(True), height=2, width=35)
button_signin = tkinter.Button(root, text="Register new user", command=sign, height=2, width=35)

# Place the elements in a grid
idneo.grid(row=0, column=0, pady=pady, padx=padx)
button_2d.grid(row=1, column=0, pady=pady, padx=padx)
button_3d.grid(row=2, column=0, pady=pady, padx=padx)
button_signin.grid(row=3, column=0, pady=pady, padx=padx)
vision.grid(row=4, column=0, pady=pady, padx=padx)

root.mainloop()
