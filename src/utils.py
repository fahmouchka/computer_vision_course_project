import os
import pandas as pd
import zipfile

import cv2
import torch
from torch.utils.data import Dataset

def generate_csv(root_folder: str, label_map: dict) -> pd.DataFrame:
    """A function that generates a dataframe with two features: image_id that corresponds to the corresponding path and the label that corresponds to the integer label.

    Args:
        root_folder (str): The folder of the images.
        label_map (dict): A dictionary that maps string labels to the corresponding integer ones.

    Returns:
        _type_: _description_
    """
    image_paths = []
    labels = []

    for foldername in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, foldername)
        if not os.path.isdir(folder_path):
            continue
        
        label = foldername
        label_num = label_map[label]  

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'): 
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                labels.append(label_num)

    data = {'image_id': image_paths, 'label': labels}
    df = pd.DataFrame(data)
    return df

def unzip(zip_file_path) -> None:
    """A function that unzips that .zip dataset file to create two seperate folders: train and test.

    Args:
        zip_file_path (_type_): _description_
    """
    extract_to = os.path.dirname(zip_file_path)

    files_exist = all(os.path.exists(os.path.join(extract_to, file)) for file in zipfile.ZipFile(zip_file_path, 'r').namelist())

    if not files_exist:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Files extracted successfully.")
    else:
        print("Files already exist, extraction not needed.")
        
        
class Dataset(Dataset):
    """Custom pytorch dataloader 
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label