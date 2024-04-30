import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import DataReader
import torch

class DataLoader(Dataset):
    def __init__(self, datasets):
        super(DataLoader).__init__()
        self.data = datasets

    def __getitem__(self, index):
        band1 = self.data[index][0]
        band2 = self.data[index][1]
        band3 = self.data[index][2]
        band4 = self.data[index][3]
        band5 = self.data[index][4]
        label = self.data[index][5]


        values =np.array([band1,
                            band2,
                            band3,
                            band4,
                          band5])

        values = values.reshape(values.shape[0])
        label = label.reshape(1)

        values = torch.tensor(values, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return values, label


    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    dataset_path = r"D:\Github_Repo\WaterDepth\Datasets\Sentinel0609"
    T,V = DataReader.Reader(dataset_path)
    train_dataloader = DataLoader(T)
    T = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=1,
        shuffle=True
    )
    test_dataloader = DataLoader(V)
    V = torch.utils.data.DataLoader(
        dataset=test_dataloader,
        batch_size=1,
        shuffle=True
    )
    for x, y in T:
        print(x.shape,y.shape)
