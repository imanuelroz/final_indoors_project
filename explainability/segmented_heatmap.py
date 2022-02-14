import cv2
import numpy as np
from matplotlib import pyplot as plt

segment_img = cv2.imread("/hdd2/airbnb_geolocation_weights/segmented.jpeg")
segment_img = cv2.resize(segment_img, (224, 224))
segment_img = np.float32(segment_img) / 255
plt.imshow(segment_img)