import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import os
import pandas as pd

transform = transforms.Compose([
    transforms.RandomRotation(30),      # rotate +/- 30 degrees
    transforms.RandomHorizontalFlip(),  # rHorizontally flip the given image randomly with a given probability (default p=0.5)
    #transforms.RandomVerticalFlip() #Vertically flip the given image randomly with a given probability (default p=0.5), not recommended for medical images
    transforms.Resize((224, 224)),       #  be sure to pass in a list or a tuple
    transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
    transforms.RandomAdjustSharpness(1.5, p=0.5), #
    transforms.RandomAdjustSharpness(0.5, p=0.5),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomEqualize(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])

class CovidSeverityDataset(Dataset):
    def __init__(self, root_dir, transform = transform, split_lengths = [0.7, 0.1, 0.2], split_seed = 42, batch_size = 10, shuffle = True, num_workers = [2,0,0]):
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
      y_label = torch.tensor(self.dataframe.iloc[index, 1])
      if self.transform:
        image = self.transform(image)
      else:
        convert_tensor = transforms.PILToTensor()
        image = convert_tensor(image)
      return (image, y_label)
  
    def get_subsets(self):
      subsets = random_split(self, self.split_lengths, generator=torch.Generator().manual_seed(self.split_seed))
      train = torch.utils.data.DataLoader(dataset=subsets[0], batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers[0])
      val = torch.utils.data.DataLoader(dataset=subsets[1], batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers[1])
      test = torch.utils.data.DataLoader(dataset=subsets[2], batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers[2])
      return train, val, test


# How to use this class?
# root_dir = "/content/drive/MyDrive/Mila/Winter_2023/ift6759_project/"
# dataset = CovidSeverityDataset(root_dir, transform = False)
# train, val, test = dataset.get_subsets()