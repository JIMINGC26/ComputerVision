import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import DatasetFromFolder
from model import SRCNN

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--expriment_num', type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

BATCH_SIZE = 4
NUM_WORKERS = 0

train_set = DatasetFromFolder("dataset/T91", zoom_factor=args.zoom_factor)
test_set = DatasetFromFolder("dataset/Set5", zoom_factor=args.zoom_factor)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(
    [
        {"params":model.conv1.parameters(), "lr":0.0001},
        {"params": model.conv2.parameters(), "lr": 0.0001},
        {"params": model.conv3.parameters(), "lr": 0.00001},
    ], lr=0.00001,
)


for epoch in range(args.num_epochs):
    # train
    epoch_loss = 0
    for iter, batch in enumerate(train_loader):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        out = model(input)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(train_loader)}")

    # test
    avg_psnr = 0
    with torch.no_grad():
        for batch in test_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            out = model(input)
            loss = criterion(out, target)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr
    
    
        print(f"Average PSNR: {avg_psnr / len(test_loader)} dB.")

        # Save model
torch.save(model, f"exp/model_{args.expriment_num}.pth")