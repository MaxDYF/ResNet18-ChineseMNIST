import torch.nn

from const_args import *
from resnet18 import *
from dataset import *
import argparse
import time
from datetime import datetime
def train():

    parser = argparse.ArgumentParser(description='CNN for FashionMNIST')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--device', help='Choose Device', type=str)
    args = parser.parse_args()
    device = torch.device('cpu')
    if args.device:
        if args.device == 'cuda':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == 'mps':
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
        elif args.device == 'xpu':
            device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    train_data = torch.load(
        args.input,
        weights_only=False
    )
    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    net = ResNet18(15).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        start = time.time()
        loss_total = 0
        step = 0
        for image, target in dataloader:
            st = time.time()
            net.zero_grad()
            step += 1
            result = net.forward(image.to(device)).to(device)
            loss = loss_fn(result, target.to(device))
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            ed = time.time() - st
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.3f}, Speed: {ed: .2f} s/it")
        end = time.time()
        elapsed = end - start
        loss_average = loss_total / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {loss_average}, Speed: {elapsed:.2f} s/it")

    nowtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output + f'/model{nowtime}.pkl'
    print(f"Saving model to {output_path}")
    torch.save(net.state_dict(), output_path)

if __name__ == '__main__':
    train()