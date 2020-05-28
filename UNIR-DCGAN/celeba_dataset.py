from __future__ import division
from __future__ import print_function

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from etc import config
import corruption

def get_dataloader(config):
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    dataroot = config["data_path"]
    workers = config["workers"]
    loss_rate = config["loss_rate"]

    tf = transform=transforms.Compose([
           transforms.Resize(image_size),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ])

    dataset = dset.ImageFolder(root=dataroot, transform=tf)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=workers)
    for img, label in dataloader:
        img = corruption.RemovePixel(img, loss_rate)

    return dataloader

dataloader = get_dataloader(config)