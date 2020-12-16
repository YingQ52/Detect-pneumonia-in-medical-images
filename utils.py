import matplotlib.pyplot as plt
import pydicom
import numpy as np

def show_image(img_path):
    plt.imshow(pydicom.read_file(img_path).pixel_array)