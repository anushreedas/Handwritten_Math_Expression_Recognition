import pandas as pd
import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


from torchvision import transforms, utils
import cv2


class data(Dataset):
    def __init__(self, data_frame, transform=None):
        self.x = data_frame['x']
        self.y = data_frame['y']
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_x = cv2.cvtColor(self.x[idx],cv2.COLOR_GRAY2RGB)
        sample_x = np.asarray(sample_x)
        sample_y = np.asarray(self.y[idx])
        if self.transform:
            sample_x = self.transform(sample_x)
        return sample_x, sample_y


# standalone changes
# check this function, it may help for the dtree too
# facing memory errors
def reformat_images(pickled_file):
    # UNCOMMENT THIS FOR THIS FOR SUBMISSION

    # df = pd.read_pickle(pickled_file)
    # uniques = list(df['labels'].unique())
    # # sanity check
    # # print(uniques)
    # classes = {}
    # for i, val in enumerate(uniques):
    #     classes[val] = i
    # uniques = None
    # # print(classes)
    # df['classes'] = df['labels'].apply(lambda x: classes[x])
    # print('Done Classes')
    # print(df['classes'])
    # classes = None
    #
    # primary, discarded = train_test_split(df, test_size=0.5)
    # df = None
    # primary.to_pickle('primary.pkl')
    # primary.to_csv('primary.csv')
    # print('Saved Primary dataset')
    # discarded.to_pickle('holdout.pkl')
    # discarded.to_csv('holdout.csv')
    # print('Saved holdout set')

    df = pd.read_pickle('primary.pkl')
    df['x'] = df['images'].apply(lambda x: x.astype('uint8'))
    print('Converted to uint8')
    df['x'] = df['images'].apply(lambda x: cv2.resize(x, (256, 256)))
    print('Resized all to 256 x 256')
    # Cant do this in place due to memory issues, hacked using dataloader
    # df['x'] = df['x'].apply(lambda x: cv2.cvtColor(x,cv2.COLOR_GRAY2RGB))
    # print('Done 2')
    train,test = train_test_split(df,test_size=0.2)
    train.to_pickle('train.pkl')
    train.to_csv('train.csv')
    print(len(train), ' : Train Samples')
    train = None
    print('Saved Train test')
    test.to_pickle('test.pkl')
    test.to_csv('test.csv')
    print(len(test), ' : Test Samples')
    test=None
    print('Saved test set')

    return None


def evaluate_data(trainset,testset):
    train_set = pd.read_pickle(trainset)
    # later
    # test_set = pd.read_pickle(testset)
    train_transofrm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    training_set = data(train_set,train_transofrm)

    train, validate, test = random_split(training_set, [30000, 10000, 10000])


def main():
    reformat_images('t1.pkl')


if __name__ == '__main__':
    main()
