import torch
def evaluate(net, data, device):
    n_total = 0
    n_accepted = 0
    with torch.no_grad():
        for image, target in data:
            results = net.forward(image.to(device)).to(device)
            for i, result in enumerate(results):
                if torch.argmax(result) == target[i]:
                    n_accepted += 1
                n_total += 1
    return n_accepted / n_total