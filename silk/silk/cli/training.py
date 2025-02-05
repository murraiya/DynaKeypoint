# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import torch
import numpy as np
from silk.config.core import instantiate_and_ensure_is_instance
from silk.config.model import load_model_from_checkpoint
from silk.logger import LOG
from torch.utils.data import DataLoader



def main(config: DictConfig):
    """Run a generic configurable training loop.

    Parameters
    ----------
    config : DictConfig
        Full configuraton structure. Important paths are described below.

    Paths
    ----------
    * `mode.model` : `LightningModule`

        Model to train.

    * `mode.loaders.training` : `DataLoader`

        Data loader used for training.

    * `mode.loaders.validation` : `DataLoader`

        Data loader used for validation.

    * `mode.trainer` : `Trainer`

        Trainer engine to run.

    Returns
    -------
    [type]
        [description]

    TODO(Pierre) : See TODO at end of function.
    """

    model = instantiate_and_ensure_is_instance(config.mode.model, pl.LightningModule)
    # here occurs silk.models.silk _init_loss_flow() 

    if config.mode.continue_from_checkpoint is not None:
        LOG.warning(
            f"the model's weight are being loaded from checkpoint : {config.mode.continue_from_checkpoint}\nplease make sure to disable it if that's not the intended behavior (by setting `config.mode.continue_from_checkpoint` to `null`)."
        )
        
        load_model_from_checkpoint(
            model,
            checkpoint_path=config.mode.continue_from_checkpoint,
            strict=False,
            freeze=False,
            eval=False,
        )



    # plt.figure(figsize=(20,5))
    # for name, param in model.named_parameters():
    #     if name.split('_')[0]=='model.model.kpt' or name.split('_')[0]=='model.model.desc':
        
    #         bins = torch.linspace(-1,1,30)
    #         # hist = [torch.histogram(c, bins = bins) for c in param.reshape(-1).detach().cpu()]
    #         hist = torch.histogram(param.detach().cpu(), bins = bins)

    #         print(len(hist))
    #         plt.plot(hist.bin_edges[:-1], hist.hist, color=np.random.rand(3,), label="{}".format(name))
    #         plt.legend()

    #         pass

    #     else: param.requires_grad = False
    # plt.savefig("./folder_for_viz/weights_xavier_50.png".format(name))

    train_loader = instantiate_and_ensure_is_instance(
        config.mode.loaders.training, DataLoader
    )
    val_loader = instantiate_and_ensure_is_instance(
        config.mode.loaders.validation, DataLoader
    )
    
    trainer = instantiate_and_ensure_is_instance(config.mode.trainer, pl.Trainer)     
    trainer.fit(model, train_loader, val_loader) #usually this becomes train
    
    # TEMPORARY(Pierre) : return some value for quick testing of pipeline
    # TODO(Pierre) : remove and replace by loss value ?
    return {
        "model": model,
        "val_loader": val_loader,
        "train_loader": train_loader,
        "trainer": trainer,
        "config": OmegaConf.to_object(config),
    }
