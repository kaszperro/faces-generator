import tkinter as tk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Evaluation import generate_image_from_vector


class App:
    def __init__(self, num_sliders):
        self.root = tk.Tk()
        self.root.title("faces generator")
        self.num_sliders = num_sliders
        self.slider_values = np.random.rand(self.num_sliders)
        self.generate_sliders()
        self.noise = None
        self.generate_noise()

        image = generate_image_from_vector(self.noise)
        f = Figure(figsize=(3, 3), dpi=100)
        a = f.add_subplot(111)
        self.face_image = a.imshow(image)
        self.canvas = FigureCanvasTkAgg(f, self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=num_sliders)

        self.generate_image()
        self.root.mainloop()

    def generate_sliders(self):
        for i in range(self.num_sliders):
            slider = tk.Scale(
                self.root,
                from_=-1.0,
                to=1.0,
                resolution=0.01,
                orient=tk.HORIZONTAL
            )
            slider.set(self.slider_values[i])
            slider.grid(row=i, column=0)
            slider.bind("<B1-Motion>", lambda x, s=slider, index=i: self.slider_val_changed(index, s))

    def slider_val_changed(self, index, slider):
        self.slider_values[index] = slider.get()
        self.generate_noise()
        self.generate_image()

    def generate_noise(self):
        self.noise = np.random.rand(100)
        for i in range(self.num_sliders):
            self.noise[i * 100 // self.num_sliders:(i + 1) * 100 // self.num_sliders] = self.slider_values[i]

    def generate_image(self):
        image = generate_image_from_vector(self.noise)
        self.face_image.set_data(image)
        self.canvas.draw()


def main():
    matplotlib.use("TkAgg")
    App(10)


if __name__ == '__main__':
    main()
