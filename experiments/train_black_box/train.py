import logging
import os

import hydra
import torch
import torchvision.models as m
from cb2.dataset.constant import CLASSES, TEST_SPLIT
from cb2.dataset.loader import HF_DATASET_TORCH
from cb2.trainer.module import Trainer_BB
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


@hydra.main(version_base=None, config_path="conf", config_name="training")
def train(cfg):
    # SEEDS
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)

    # LOGGER
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    
    # LOAD BLACK BOX
    if cfg.black_box=="resnet18":
        bb_weights = ResNet18_Weights.DEFAULT
        bb_model = resnet18(weights=bb_weights,
                            progress=False).eval().to(cfg.device)
        bb_model.requires_grad_(False)
        bb_model.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=64, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=.5, inplace=False),
                torch.nn.Linear(in_features=64, out_features=CLASSES[cfg.dataset], bias=True),
                )
        bb_model.fc.requires_grad_(True)

    elif cfg.black_box=="vgg16":
        bb_weights = m.VGG16_Weights.DEFAULT
        bb_model  = m.vgg16(weights=bb_weights,
                progress=False).to(cfg.device)
        bb_model.requires_grad_(False)
        bb_model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=.5, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=.5, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=512, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=.5, inplace=False),
                torch.nn.Linear(in_features=512, out_features=512, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=.5, inplace=False),
                torch.nn.Linear(in_features=512, out_features=CLASSES[cfg.dataset], bias=True),
                )
        bb_model.classifier.requires_grad_(True)
    elif cfg.black_box=="vitB16":
        bb_weights = m.ViT_B_16_Weights.DEFAULT
        bb_model = m.vit_b_16(weights=bb_weights,
                progress=False).to(cfg.device)
        bb_model.requires_grad_(False)
        in_head = bb_model.heads.head.in_features
        bb_model.heads = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.75),
                    torch.nn.Linear(in_features=in_head, out_features=in_head//2, bias=True),
                    torch.nn.SiLU(),
                    torch.nn.Linear(in_features=in_head//2, out_features=CLASSES[cfg.dataset], bias=True),
                    )
        bb_model.heads.requires_grad_(True)

    else:
        logger.error(f"Unknown black box {cfg.black_box}")
    bb_transform = bb_weights.transforms()
    # DATASET DIRECTORY
    dataset_train = HF_DATASET_TORCH(
            data_dir = cfg.path_raw_dataset,
            dataset = cfg.dataset,
            split='train',
            transform = bb_transform
            )
    dataset_test = HF_DATASET_TORCH(
            data_dir = cfg.path_raw_dataset,
            dataset = cfg.dataset,
            split=TEST_SPLIT[cfg.dataset],
            transform = bb_transform
            )
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=cfg.batch_size,
                                  drop_last=True,
                                  shuffle=True)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=cfg.batch_size,
                                 drop_last=True,
                                 shuffle=True)
 
    # LOAD TRAINER
    trainer = Trainer_BB(device=cfg.device,
                            model=bb_model,
                            transform=bb_transform,
                            lr=cfg.lr,
                            r_pred=cfg.r_pred,
                            training_ablation=cfg.training_ablation,
                            )
    logger.info("Launching the training...")
    trainer.fit(data=dataloader_train,
                test_data=dataloader_test,
                epochs=cfg.epochs)
    torch.save(trainer.model,
               os.path.join(trainer.save_path, "resnet18.pk"))
    torch.save(
        trainer.model.state_dict(),
        os.path.join(trainer.save_path, "resnet18_state.pk"),
    )
    logger.info(f"Model saved at {trainer.save_path}")


if __name__=='__main__':
    train()
