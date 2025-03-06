import argparse

import torch
from sklearn.model_selection import train_test_split
from dataset import *
def prepare(csv_src, data_dir_src, save_dir_src):
    images, targets = load_data(csv_src, data_dir_src)
    train_img, test_img, train_tar, test_tar = train_test_split(images, targets, test_size=0.25, random_state=42)
    train_data = MnistDataset(train_img, train_tar)
    test_data = MnistDataset(test_img, test_tar)
    torch.save(train_data, f'{save_dir_src}/train_data.pt')
    torch.save(test_data, f'{save_dir_src}/test_data.pt')
    print('Data prepared.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_src', type=str, default='data/mnist/mnist_train.csv')
    parser.add_argument('--data', type=str)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    prepare(args.csv_src, args.data, args.save)