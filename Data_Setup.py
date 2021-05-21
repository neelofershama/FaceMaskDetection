from __future__ import print_function, division

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from sklearn.model_selection import KFold

def load_data(datasetpath):
    maskPath = datasetpath/'WithMask'
    nonMaskPath = datasetpath/'WithoutMask'
    randomPath = datasetpath/'Random'
    maskDF = pd.DataFrame()

    for imgPath in tqdm(list(maskPath.iterdir()), desc='Loading with mask data'):
        maskDF = maskDF.append({
            'image': str(imgPath),
            'class': 0
        }, ignore_index=True)

    for imgPath in tqdm(list(nonMaskPath.iterdir()), desc='Loading without mask data'):
        maskDF = maskDF.append({
            'image': str(imgPath),
            'class': 1
        }, ignore_index=True)

    for imgPath in tqdm(list(randomPath.iterdir()), desc='Loading random images data'):
        maskDF = maskDF.append({
            'image': str(imgPath),
            'class': 2
        }, ignore_index=True)

    dfName = datasetpath/'dataset.pickle'
    print(f'Saving Dataframe to: {dfName}')
    maskDF.to_pickle(dfName)
    return dfName


class MaskDetectionDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.transformations = Compose(
            [Resize((32, 32)),
             ToTensor(),
             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]
        image = Image.open(row['image'])
        return {
            'image': self.transformations(image),
            'class': tensor([row['class']], dtype=long),
        }

    def __len__(self):
        return len(self.dataFrame.index)

def prepare_data_kfold(pickle_path):
    data_df = pd.read_pickle(pickle_path)
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    train=[]
    test =[]
    for train_i , test_i in kf.split(data_df):
        train_data=data_df.iloc[train_i]
        test_data=data_df.iloc[test_i]
        train_set = MaskDetectionDataset(train_data)
        test_set = MaskDetectionDataset(test_data)
        train.append([train_set])
        test.append(test_set)
    return [train,test]


def prepare_data(pickle_path) -> None:
    data_df = pd.read_pickle(pickle_path)
    train_data, validate_data = train_test_split(data_df, test_size=0.3, random_state=0,
                     stratify=data_df['class'])

    train_set = MaskDetectionDataset(train_data)
    test_set = MaskDetectionDataset(validate_data)
    return [train_set, test_set]

def load_prepare_test_image(path):
    folder = list(path.split("/"))[-2]
    if folder== "WithMask":
        c =0
    elif folder== "WithoutMask":
        c=1
    else:
        c =2
    data_frame = pd.DataFrame()
    data_frame = data_frame.append({
        'image': str(path),
        'class': c
    }, ignore_index=True)
    test_pickle = "Face Mask Dataset/Test/test.pickle"
    data_frame.to_pickle(test_pickle)

    test_image = MaskDetectionDataset(data_frame)
    return test_image