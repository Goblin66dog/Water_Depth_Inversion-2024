import glob
import os.path

import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import Valid_fig
import DataReader
import DataLoader
from model import MLP


def test(model_path):
    # file_path = "test logs.txt"
    # file = open(file_path, "w", encoding="utf-8")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets_path = r"D:\Github_Repo\WaterDepth\Datasets\Landsat0607"
    T, V = DataReader.Reader(datasets_path)
    valid_dataloader = DataLoader.DataLoader(V)

    V = torch.utils.data.DataLoader(
        dataset=valid_dataloader,
        batch_size=1,
        shuffle=False
    )

    model = MLP(in_channels=5, out_channels=1, hidden_size=1024)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载pth文件
    model.to(device=device)
    model = model.eval()

    label = []
    prediction = []
    for val_value, val_label in V:
        model.eval()
        with torch.no_grad():
            val_value = val_value.to(device=device, dtype=torch.float32)
            val_label = val_label.to(device=device, dtype=torch.float32)
            val_prediction = model(val_value)
            val_prediction = np.array(val_prediction.data.cpu()[0][0])
            val_label = np.array(val_label.data.cpu()[0][0])
            # print(val_prediction, val_label)
            label.append(val_label)
            prediction.append(val_prediction)

    label = np.array(label)
    prediction = np.array(prediction)

    Valid_fig.scat_fig(label, prediction,"d")


    mse = mean_squared_error(label, prediction)
    mae = mean_absolute_error(label, prediction)
    mape = mean_absolute_percentage_error(label, prediction)
    r2   = r2_score(label, prediction)
    return mse, mae, mape, r2

model_path_list = glob.glob(os.path.join(r"D:\Github_Repo\MLP_WaterDepth",r"*.pth"))
for each_path in model_path_list:
    mse, mae, mape, r2 = test(each_path)
    name = os.path.splitext(os.path.basename(each_path))[0]
    print("#########"+str(name)+"###########")
    print("mse:",mse)
    print("mae:",mae)
    print("mape:",mape)
    print("r2:", r2)
