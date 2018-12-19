from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from obj.devices import Camera


camera_device = Camera(0)
def clicked():
    print(str(combo.get()))
    return combo.get()

def callback(eventObject):
    # you can also get the value off the eventObject
    print(eventObject.widget.get())
    # to see other information also available on the eventObject
    print(dir(eventObject))

num_camera = 4

window = Tk()
window.title("Camera View finder")
window.geometry('450x450')

data = camera_device.getFrame()
img = Image.fromarray(data, 'RGB')

img_tk = ImageTk.PhotoImage(img)

panel = Label(window, image = img_tk)
panel.pack(side = "bottom", fill = "both", expand = "yes")

#lbl = Label(window, text="Select the camera: ")
#lbl.grid(column=0, row=0)

btn = Button(window, text="View", command=clicked)
#btn.grid(column=3, row=0)

combo = Combobox(window)
#combo['values'] = ['Camera %s' % cam for cam in range(num_camera)]

#combo.grid(column=2, row=0)

window.mainloop()


