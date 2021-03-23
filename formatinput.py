import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import cv2


class data(Dataset):
    def __init__(self, data_frame, transform=None):
        self.x = list(data_frame['x'])
        self.y = list(data_frame['classes'])
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_x = Image.fromarray(np.asarray(self.x[idx]))
        sample_x = sample_x.convert('RGB')
        sample_y = self.y[idx]
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


def evaluate_data(trainset,testset=None):
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    train_set = pd.read_pickle(trainset)
    train_set, test = train_test_split(train_set, test_size=0.6)

    # later
    # test_set = pd.read_pickle(testset)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    training_set = data(train_set,train_transform)
    print(training_set.__len__())
    train, validate, test = random_split(training_set, [10000, 0, 3728])

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)

    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Sequential(nn.Linear(in_features=4096, out_features=101),
                                         nn.LogSoftmax(dim=1))
    model.to(device)
    print(model)
    optimizer = Adam(model.parameters())
    loss_criterion = nn.NLLLoss()
    epochs = 10
    batch_loss = 0
    total_epoch_loss = 0
    history = []
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        with torch.no_grad():

            model.eval()

            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j,
                                                                                                           loss.item(),
                                                                                                           acc.item()))

        avg_train_loss = train_loss / 10000
        avg_train_acc = train_acc / float(10000)

        avg_valid_loss = valid_loss / 3728
        avg_valid_acc = valid_acc / float(3728)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))

    history = np.array(history)
    print("HISTORY", history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('_loss_curve.png')
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('_accuracy_curve.png')
    plt.show()

















def main():
    # reformat_images('t1.pkl')
    train = '/home/omkarsarde/PycharmProjects/Datasets/train.pkl'
    evaluate_data(train)

if __name__ == '__main__':
    main()
