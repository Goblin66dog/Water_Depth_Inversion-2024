import glob
import os.path

import pandas as pd
import numpy as np


def Reader(dataset_path):
    dataset_path = glob.glob(os.path.join(dataset_path, "*"))
    train_datasets = "None"
    test_datasets  = "None"
    for each_dataset_path in dataset_path:
        if "train" in each_dataset_path:
            train_datasets= pd.read_csv(each_dataset_path)
        if "test"  in each_dataset_path:
            test_datasets = pd.read_csv(each_dataset_path)
    #############加载训练样本
    train_dataset = []
    for sample_num in range(len(train_datasets["water_depth"])):
        B1      = train_datasets["band1_Coastal Aerosol"]  [sample_num]
        B2      = train_datasets["band2_Blue"] [sample_num]
        B3      = train_datasets["band3_Green"]   [sample_num]
        B4      = train_datasets["band4_Red"]   [sample_num]
        B5      = train_datasets["band5_NIR"]   [sample_num]

        label   = train_datasets["water_depth"] [sample_num]
        train_dataset.append([B1,B2,B3,B4,B5,label])
    train_datasets = np.array(train_dataset, dtype=np.float32)
    del train_dataset
    #############加载测试样本
    test_dataset = []
    for sample_num in range(len(test_datasets["water_depth"])):
        B1      = test_datasets["band1_Coastal Aerosol"]  [sample_num]
        B2      = test_datasets["band2_Blue"] [sample_num]
        B3      = test_datasets["band3_Green"]   [sample_num]
        B4      = test_datasets["band4_Red"]   [sample_num]
        B5      = test_datasets["band5_NIR"]   [sample_num]

        label   = test_datasets["water_depth"] [sample_num]
        test_dataset.append([B1,B2,B3,B4,B5,label])
    test_datasets = np.array(test_dataset, dtype=np.float32)
    del test_dataset
    return train_datasets, test_datasets




# for values in range(len(datasets["water_depth"])):
#
if "__main__" == __name__:
    dataset_path = r"D:\Github_Repo\WaterDepth\Datasets\Landsat0608"
    T, V = Reader(dataset_path)
    print(T.shape)