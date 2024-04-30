import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import Data_Reader_Array
import DataReader
import DataLoader
from model import MLP
from osgeo import gdal

def SaveWithGeoInfo(item, image):
    axs = [0, 1], [0, 1]

    image = np.transpose(image, axs[0])
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create("prediction.TIF",
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

def deploy(model_path,image_path):
    # file_path = "test logs.txt"
    # file = open(file_path, "w", encoding="utf-8")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_item = Data_Reader_Array.Dataset(image_path)
    image = image_item.array
    image = np.transpose(image,[1,2,0])
    image = torch.tensor(image, dtype=torch.float32)
    model = MLP(in_channels=5, out_channels=1, hidden_size=1024)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载pth文件
    model.to(device=device)
    model = model.eval()

    output = []
    for rows in image:
        # for cols in rows:
        rows = rows.reshape(1256,5)
        rows = rows.to(device=device, dtype=torch.float32)
        rows = model(rows)
        rows = np.array(rows.data.cpu())
        rows = np.transpose(rows, [1,0])[0]
        output.append(rows)
    output = np.array(output)
    mat = image_item.array[0]
    mat[mat > 0] = 1
    # plt.figure()
    # plt.imshow(mat)
    # plt.show()
    output = (-output)*mat
    output += 716

    # SaveWithGeoInfo(image_item, output)
    plt.figure()
    heatmap=plt.imshow(output,"coolwarm")
    plt.colorbar(heatmap)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(r"C:\Users\Vitch\Desktop\waterdepth.png",dpi=300,bbox_inches='tight')

if "__main__" == __name__:
    data_path = r"D:\Github_Repo\WaterDepth\Datasets\lake.tif"
    model_path = \
        r"D:\Github_Repo\WaterDepth\logs\LandSat\logs1\Chosen.pth"
    deploy(model_path,data_path)