import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import os
import pandas as pd

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
  transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
  transforms.ToTensor(),
  transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
])

class CovidSeverityDataset(Dataset):
    def __init__(self, root_dir, transform = transform, split_lengths = [0.7, 0.1, 0.2], split_seed = 42, batch_size = 64, shuffle = True, num_workers = [0,0,0]):
      self.root_dir = root_dir
      self.csv_path = self.root_dir + "data_processing/combined_cxr_metadata.csv"
      self.data_path = self.root_dir + "processed_images/"
      self.transform = transform
      self.split_lengths = split_lengths
      self.split_seed = split_seed
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.num_workers = num_workers
      self.dataframe = pd.read_csv(self.csv_path, index_col=0)
  
    def __len__(self):
      return len(self.dataframe)

    def __getitem__(self, index):
      img_path = os.path.join(self.data_path, self.dataframe.iloc[index, 0])
      image = Image.open(img_path)
      y_label = torch.round(torch.tensor(self.dataframe.iloc[index, 1]))
      if self.transform:
        image = self.transform(image)
      else:
        convert_tensor = transforms.PILToTensor()
        image = convert_tensor(image)
        image = image.float()
      return (image, y_label.to(torch.int64))
  
    def get_subsets(self):
      subsets = random_split(self, self.split_lengths, generator=torch.Generator().manual_seed(self.split_seed))
      train = torch.utils.data.DataLoader(dataset=subsets[0], batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers[0], pin_memory=True)
      val = torch.utils.data.DataLoader(dataset=subsets[1], batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers[1])
      test = torch.utils.data.DataLoader(dataset=subsets[2], batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers[2])
      return train, val, test