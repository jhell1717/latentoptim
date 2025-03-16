import torch
from torch.utils.data import Dataset


class ShapeData(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self,shapes):
        self.shapes = shapes

    def __len__(self):
        return len(self.shapes)
    
    def __getitem__(self,idx):
        shape = self.shapes[idx]
        return torch.tensor(shape.points,dtype=torch.float32).view(-1)