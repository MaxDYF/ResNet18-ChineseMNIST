import pandas
import cv2
from torchvision.transforms import transforms
import torch
def load_data(csv_src, img_dir_src):
    table = pandas.read_csv(csv_src)
    transform = transforms.ToTensor()
    x = []
    y = []
    for index, row in table.iterrows():
        image_name = f"input_{row[0]}_{row[1]}_{row[2]}.jpg"
        img = cv2.imread(f"{img_dir_src}/{image_name}")
        upscale_img = cv2.resize(img, (224, 224))
        img_tensor = transform(upscale_img).float()
        x.append(img_tensor)
        y.append(row[2] - 1)
    return (torch.stack(x, dim=0),
           torch.tensor(y, dtype=torch.int64))


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, images, target):
        super().__init__()
        self.images, self.targets = images, target
    def __getitem__(self, index):
        return self.images[index], self.targets[index]
    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    x, y = load_data("./archive/chinese_mnist.csv", "./archive/data/data")