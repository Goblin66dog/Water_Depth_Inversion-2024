import warnings
import random

import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import recall_score, precision_score, mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import DataReader
import DataLoader
from model import MLP
import time
def train(device, epochs,batch_size,lr,datasets_path):
    batch_size_str = batch_size
    # file_path = "valid logs.txt"
    # file = open(file_path, "w", encoding="utf-8")
    # file.writelines("      " +"value" + "   " + "label" + "\n")
    timestamp = time.time()
    readable_date_time = time.ctime(timestamp).replace(":", "_")
    writer = {
        "loss": SummaryWriter("logs"+"\\("+
                              "lr-"+str(lr)+
                              " epochs-"+str(epochs)+
                              " batch_size-"+str(batch_size)+")"+
                              str(readable_date_time)+" loss"),
        "mse": SummaryWriter("logs" + "\\(" +
                              "lr-" + str(lr) +
                              " epochs-" + str(epochs) +
                              " batch_size-" + str(batch_size) + ")" +
                              str(readable_date_time) + "mse"),
        "mae": SummaryWriter("logs" + "\\(" +
                             "lr-" + str(lr) +
                             " epochs-" + str(epochs) +
                             " batch_size-" + str(batch_size) + ")" +
                             str(readable_date_time) + "mae"),
        "mape": SummaryWriter("logs" + "\\(" +
                             "lr-" + str(lr) +
                             " epochs-" + str(epochs) +
                             " batch_size-" + str(batch_size) + ")" +
                             str(readable_date_time) + "mape"),
        "r2": SummaryWriter("logs" + "\\(" +
                             "lr-" + str(lr) +
                             " epochs-" + str(epochs) +
                             " batch_size-" + str(batch_size) + ")" +
                             str(readable_date_time) + "r2"),

    }

    #网络加载
    net = MLP(in_channels=5, out_channels=1, hidden_size=1024)
    net.to(device=device)
    net.train()

    #数据加载
    T,V = DataReader.Reader(datasets_path)
    train_dataloader = DataLoader.DataLoader(T)
    T = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=batch_size,
        shuffle=True
    )
    valid_dataloader = DataLoader.DataLoader(V)
    V = torch.utils.data.DataLoader(
        dataset=valid_dataloader,
        batch_size=1,
        shuffle=True
    )

    #loss
    loss_function = nn.SmoothL1Loss()
    best_loss = float("inf")

    #optimizer
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):

        # batch_size = batch_size*9//10
        # if  batch_size < 4:
        #     batch_size = 1
        # if epoch == epochs//2:
        #     batch_size =batch_size//2
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        # elif epoch == epochs*2//3:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        #     batch_size = 1
        for value, label in T:
            optimizer.zero_grad()

            value = value.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            prediction = net(value)

            loss = loss_function(prediction, label)
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), readable_date_time +
                           "(lr-" + str(lr) +
                           " epochs-" + str(epochs) +
                           " batch_size-" + str(batch_size_str) + ")"
                                                                  "model.pth")
            loss.backward()
            optimizer.step()
            step+=1

            writer['loss'].add_scalar("data", loss, step)

        if epoch % 50 == 0:
            valid_num = 0
            label = []
            prediction = []
            for val_value, val_label in V:
                net.eval()
                with torch.no_grad():
                    if valid_num == 50:
                        net.train()
                        break
                    val_value = val_value.to(device=device, dtype=torch.float32)
                    val_label = val_label.to(device=device, dtype=torch.float32)
                    val_prediction = net(val_value)
                    val_prediction = np.array(val_prediction.data.cpu()[0][0])
                    val_label = np.array(val_label.data.cpu()[0][0])
                    label.append(val_label)
                    prediction.append(val_prediction)
                    valid_num += 1
            label = np.array(label)
            prediction = np.array(prediction)
            mse = mean_squared_error(label, prediction)
            mae = mean_absolute_error(label, prediction)
            mape = mean_absolute_percentage_error(label, prediction)
            r2 = r2_score(label, prediction)
            writer['mse'].add_scalar("data", mse, step)
            writer['mae'].add_scalar("data", mae, step)
            writer['mape'].add_scalar("data", mape, step)
            writer['r2'].add_scalar("data", r2, step)


if __name__ == "__main__":
    datasets_path = r"D:\Github_Repo\WaterDepth\Datasets\Landsat0607"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 6e-5
    # for times in range(6):
    train(device, 12000, 128, lr, datasets_path)
