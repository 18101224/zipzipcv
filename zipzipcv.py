import numpy as np
import cv2 as cv
img_file = './sunny.jpeg'
img = cv.imread(img_file)
#for display the original picture
import matplotlib.pyplot as plt
plt.title('The original image')
plt.imshow(img)
plt.show()
from sklearn.cluster import KMeans
img_flatten = img / 255
img_flatten = img_flatten.reshape(-1,3)
img_flatten.shape
zipper = KMeans(n_clusters=4)
zipper.fit(img_flatten)
zipped_pixel_label = zipper.predict(img_flatten)
zipped_img = zipper.cluster_centers_[zipped_pixel_label].reshape(img.shape)
plt.imshow(zipped_img)
cv.imshow('before and after',np.hstack(img,zipped_img))
cv.waitKey()
cv.destroyWindow()