import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from numpy import log
from sklearn.metrics import mean_squared_error, r2_score


def scat_fig(y_true,y_pred,title):
    """
    作散点图输出精度
    :param y_pred: 模型预测出的水深值
    :param y_true: 实际水深值
    :param title: a or other
    :return:
    """
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 计算决定系数（R²）
    r_squared = r2_score(y_true, y_pred)
    # 计算相对均方根误差（RMRSE）
    rmrse = rmse / (np.mean(y_true) - np.min(y_true)) if np.mean(y_true) - np.min(y_true) != 0 else np.nan
    # 计算平均绝对误差（MAE）
    mae = np.mean(np.abs(y_true - y_pred))
    # 计算平均绝对百分比误差（MAPE）
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
    ax.plot(y_true, y_pred, 'o', c='blue', markersize=2)
    bbox = dict(boxstyle="round", fc='1', alpha=0.)
    bbox = bbox
    plt.text(0.05, 0.65, "$R^2: %.2f$\n$RMSE: %.2f$\n$RMRSE: %.2f$\n$MAE: %.2f$\n$MAPE: %.2f$" % ((r_squared),(rmse),(rmrse),mae,mape),
             transform=ax.transAxes, size=7, bbox=bbox,fontdict={'family': 'Times New Roman','weight': 'bold'})
    plt.text(9,0.2,"(%s)" % (title))
    ax.set_xlabel('ICESat-2 depth /m', fontsize=7,fontdict={'family': 'Arial','size': 14,'weight': 'bold'})
    ax.set_ylabel("Predicted depth /m", fontsize=7,fontdict={'family': 'Arial','size': 14,'weight': 'bold'})
    ax.set(xlim=(0, 10), ylim=(0, 10))
    plt.xticks([0,2,4,6,8,10])
    plt.yticks([0,2,4,6,8,10])
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"C:\Users\Vitch\Desktop\Valid_fig.png", dpi=300)