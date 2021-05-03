from torch.utils.data import Dataset
from skimage import io

class GroceryDataset(Dataset):
    """
    The Torch Freiburg Grocery Dataset 
    """
    def __init__(self, df, label_map, transform):
        super().__init__()
        self.df = df
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        index, img_path, label = self.df.iloc[idx]
        img = io.imread(img_path)
        if self.transform:
            img = self.transform(img)

        return img, self.label_map[label]
