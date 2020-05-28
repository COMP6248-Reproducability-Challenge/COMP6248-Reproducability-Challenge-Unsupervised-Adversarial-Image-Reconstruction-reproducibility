from __future__ import division
from __future__ import print_function

import celeba_dataset
from etc import config
from graph import NNGraph

def run():
    dataloader = celeba_dataset.get_dataloader(config)
    g = NNGraph(dataloader, config)
    g.train()

run()