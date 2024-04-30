import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Data_Reader_Array

image_item = Data_Reader_Array.Dataset(r"D:\Remote Sensing Software\Workflow\星湖杯\YOUR_NEW_TIF_IMAGE_PATH.tif")
image = image_item.array
clip_image = 0

clip_image_3d = np.zeros([image.shape[1], 720])

start = 550
end = 800
# for threshold in range(start,end):
#     clip_image = image[threshold]
#     clip_image = cv2.normalize(clip_image,None, 0, 719, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
#     pixel_num = 0
#     for each_pixel in clip_image:
#         clip_image_3d[pixel_num][each_pixel] = 1
#         pixel_num += 1
#
clip_image = image[start:end]
clip_image = np.transpose(clip_image, [1,0])

df = pd.DataFrame(data=clip_image, columns=list(range(start,end)))
df.to_csv("result3d.csv", index=False)



# x_axis =np.array(list(range(image.shape[1])))
# print(x_axis.shape, clip_image_3d.shape)
# fit = np.polyfit(x_axis,clip_image_3d,20)
# x_axis_fit = np.linspace(x_axis.min(), x_axis.max(),x_axis.shape[0])
# plt.plot(x_axis_fit, clip_image_3d)
# plt.figure()
# plt.imshow(clip_image_3d)
# plt.show()