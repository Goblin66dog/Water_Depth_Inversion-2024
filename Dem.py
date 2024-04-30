import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal

import Data_Reader_Array


def SaveWithGeoInfo(item, image):
    axs = [0, 1], [0, 1]

    image = np.transpose(image, axs[0])
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create("3Dpadding.TIF",
                            image.shape[1],
                            image.shape[0],
                            1,
                            gdal.GDT_Float32)
    image = np.transpose(image, axs[1])
    dataset.SetGeoTransform(item.geotrans)  # 写入仿射变换参数
    dataset.SetProjection(item.proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(image)
    dataset.FlushCache()  # 确保所有写入操作都已完成
    dataset = None


image_item = Data_Reader_Array.Dataset(r"D:\Remote Sensing Software\Workflow\星湖杯\YOUR_NEW_TIF_IMAGE_PATH.tif")
image = image_item.array
T = 51
pad_size = int((T-1)/2)
h = image.shape[0]
w = image.shape[1]
kernel = np.ones([T])

image = np.transpose(image, [1,0])

outputw = np.ones([w,h])
for cols in range(w):
    col = np.pad(image[cols], pad_size)
    for pixel in range(pad_size, col.shape[0]-pad_size):
        ave = col[pixel-pad_size:pixel+pad_size+1]
        ave = np.mean(ave)
        outputw[cols][pixel-pad_size] = ave

outputw = np.transpose(outputw, [1,0])
outputh = np.ones([h,w])
for rows in range(h):
    row = np.pad(outputw[rows], pad_size)
    for pixel in range(pad_size, row.shape[0]-pad_size):
        ave = row[pixel-pad_size:pixel+pad_size+1]
        ave = np.mean(ave)
        outputh[rows][pixel-pad_size] = ave

outputh= np.pad(outputh, ((0,0),(100,0)),constant_values=(outputh.max(),0))
outputh = 716 -outputh

SaveWithGeoInfo(image_item, outputh)

# for rows in range(h):
#     for cols in range(w):
#