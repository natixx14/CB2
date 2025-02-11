import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import hydra
import torch


class Trainer(ABC):

    def __init__(self, model, device, lr, training_ablation=1):
        # ATTRIBUTES
        self.model = model
        self.device = device
        self.lr = lr
        self.training_ablation = training_ablation
        # LOGGING CONF
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=0,
        )
        # HISTORY LOGGER
        self.history_init()
        # Hydra handling
        self.hydra_handling()
        # SEND MODEL TO DEVICE
        self.model.to(self.device)
        # SOME VERBOSE LOG
        self.logger.info(
            f"nb parameters to learn: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        self.logger.info(f"saving path: {self.save_path}")

    def hydra_handling(self):
        try:
            self.save_path = (hydra.core.hydra_config.HydraConfig.get()["runtime"]
                          ["output_dir"] + "/")
        except Exception:  
            self.save_path = f"./tests_outputs/{datetime.now():%Y-%m-%d_%H%M%S}"
            self.logger.error(f'Unable to load hydra core output dir')
        self.logger.info(f"saving path: {self.save_path}")

    def fit_verbose(self, key_list, ep, epochs):
        if not key_list:
            key_list = self.history.keys()
        if not ep:
            self.logger.info(f"=== Epoch {ep+1}/{epochs}")
        else:
            self.logger.info(
                    f'=== Epoch {ep+1}/{epochs} '
                    + 
                    ' '.join([f'{k} : {self.history[k][-1]:.2f}' for k in key_list])
                    )
    
    def pre_fit_routine(self):
        pass

    def fit(self, data, epochs, key_list=None, test_data=None, verbose=True):
        self.pre_fit_routine()
        for ep in range(epochs):
            self.model.train()
            if verbose:
                self.fit_verbose(key_list, ep, epochs)
            # RUNNING LOSSES
            self.init_running_losses()
            # LOSS INSTANCE
            self.epoch(data)
            if test_data:
                self.model.eval()
                self.init_running_losses()
                self.epoch(test_data, train=False)
        with open(os.path.join(self.save_path, "history.pk"), "wb") as handle:
            pickle.dump(self.history, handle)
        with open(os.path.join(self.save_path, "history_test.pk"), "wb") as handle:
            pickle.dump(self.history_test, handle)
        self.logger.info("End of fit routine")

     
    @abstractmethod
    def init_running_losses(self):
        pass

    @abstractmethod
    def history_init(self):
        pass

    @abstractmethod
    def write_history(self):
        pass

    @abstractmethod
    def epoch(self, data, train=True):
        pass

