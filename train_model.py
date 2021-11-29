import os

from matplotlib.pyplot import plot
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import CenterCrop
from data import MultiFolderDataset
from physics import SobelFilter, constitutive_constraint
from unet import TurbNetG, UNet
from models import DenseED
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from plot import plot_comparison, plot_multi_comparison
from torchvision.transforms import RandomCrop, Resize, Compose, RandomRotation
import torch
import numpy as np
import random

data_path = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/generated/SingleDirection"
model_dir = "runs/run"

folder_list = [os.path.join(data_path, f"batch{i+1}") for i in range(2, 7)]
mf_dataset = MultiFolderDataset(folder_list)

train_size = int(0.8 * len(mf_dataset))
test_size = len(mf_dataset) - train_size
train_dataset, test_dataset = random_split(mf_dataset, [train_size, test_size])

rand_rot_trans = RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
crop_trans = CenterCrop(45)
resize_trans = Resize((64, 64))
trans = Compose([rand_rot_trans, crop_trans, resize_trans])

print(f"Total Length: {len(mf_dataset)}")
print(f"Test size: {len(test_dataset)}")

train_data_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, pin_memory=True
)
test_data_loader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=True, pin_memory=True
)

device = "cuda:0"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(model_dir + "_turbnet_rot_lr" + timestamp, flush_secs=10)

# model = DenseED(
#     in_channels=2,
#     out_channels=1,
#     imsize=64,
#     blocks=[3, 4, 3],
#     growth_rate=24,
#     init_features=32,
#     drop_rate=0.2,
#     out_activation=None,
#     upsample="nearest",
# )

# model = UNet(in_channels=2, out_channels=1)

model = TurbNetG(channelExponent=3)

model.to(device)

n_epochs = 6000 * 6 * 5
lr = 1e-3
batch_size = 10
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(train_data_loader), epochs=n_epochs
)

loss_fn = MSELoss()
sobel_filter = SobelFilter(64, correct=True, device=device)

postfix_dict = {
    "loss": "",
    "t_loss": "",
    "lr": "NaN",
    "pde": "NaN",
    "dir": "NaN",
    "neu": "NaN",
}
progress_bar = tqdm(desc="Epoch: ", total=n_epochs, postfix=postfix_dict, delay=0.5)

# writer.add_graph(model)


def augment_data(input, target):
    seed = np.random.randint(2147483647)  # make a seed with numpy generator
    random.seed(seed)  # apply this seed to img tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    input = trans(input)

    random.seed(seed)  # apply this seed to target tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    target = trans(target)

    return input, target


loss = 0
for epoch in range(n_epochs):
    for batch_idx, sample in enumerate(train_data_loader):
        # sample = sample.to(device)
        input = sample[0].to(device)
        target = sample[1].to(device)
        input, target = augment_data(input, target)

        # model.zero_grad()
        input.requires_grad = True
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(
            output, target
        )  # + constitutive_constraint(input, output, sobel_filter)
        loss.backward()
        optimizer.step()
        scheduler.step()

    if epoch % 100 == 0:
        model.eval()
        for test_batch_idx, test_sample in enumerate(test_data_loader):
            test_input = test_sample[0].to(device)
            test_target = test_sample[1].to(device)
            test_input, test_target = augment_data(test_input, test_target)

            test_output = model(test_input)
            test_loss = loss_fn(test_output, test_target)

        postfix_dict["t_loss"] = f"{test_loss:.5f}"
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_figure(
            "comp",
            plot_multi_comparison(test_input, test_output, test_target),
            epoch,
        )

        model.train()

    writer.add_scalar("loss", loss, epoch)
    postfix_dict["loss"] = f"{loss:.5f}"
    postfix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.5f}"
    progress_bar.set_postfix(postfix_dict)
    progress_bar.update(1)
