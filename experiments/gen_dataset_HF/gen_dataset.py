import hashlib
import logging
import os
import copy
import pickle

import hydra
import torch
from cb2.dataset.constant import TEST_SPLIT, ID_LATENT
from cb2.dataset.loader import HF_DATASET_TORCH, load_black_box, load_concepts
from datasets import load_from_disk
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


@hydra.main(config_path="conf", config_name="generator")
def generate_data(cfg):
    # LOGGER
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.info(f'device: {cfg.device}')
    if cfg.black_box not in ["resnet18", "vgg16", "vitB16"]:
        logger.error(f"{cfg.black_box} unknown")        
        return

    # LOAD BLACK BOX TEACHER
    logger.info(f'Loading black box model {cfg.black_box} for dataset {cfg.dataset}')
    bb_model, bb_transform = load_black_box(cfg.dataset, cfg.black_box,
                                            cfg.device, cfg.bb_dir)
    #bb_latent = torch.nn.Sequential(*(list(bb_model.children())[:-1]) , *(list(bb_model.children())[-1][:ID_LATENT[cfg.black_box]]))
    bb_latent = copy.deepcopy(bb_model)
    if cfg.black_box == "vitB16":
        bb_latent.heads = torch.nn.Sequential(*(list(bb_model.heads.children())[:ID_LATENT[cfg.black_box]]))
    elif cfg.black_box == "vgg16":
        bb_latent.classifier = torch.nn.Sequential(*(list(bb_model.classifier.children())[:ID_LATENT[cfg.black_box]]))
    else:
        bb_latent = torch.nn.Sequential(*(list(bb_model.children())[:-1]))
    bb_latent.requires_grad_(False)
    bb_model.requires_grad_(False)

    # LOAD CONCEPT DICO
    logger.info("Loading concept dictionnary...")
    concepts = load_concepts(concepts_bank_dir=cfg.path_dataset_dico, concepts_dico=cfg.concepts) 

    # LOAD CLIP MODEL
    clip_model = CLIPModel.from_pretrained(cfg.clip_model).to(cfg.device)
    clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model)

    # CREATE SAVE DIRECTORY
    path_to_save = os.path.join(
        cfg.data_dir, cfg.dataset, cfg.clip_model,
        hashlib.sha256(str.encode(''.join(concepts))).hexdigest(),
        cfg.black_box)
    os.makedirs(path_to_save, exist_ok=True)

    def sim_score(instance):
        inputs = clip_processor(text=concepts,
                                images=instance['image'],
                                return_tensors='pt',
                                padding=True).to(cfg.device)
        clip_outputs = clip_model(**inputs)

        # SOME PRE-PROCESSING
        l_tensor = []
        tensorTr = transforms.ToTensor()
        for i in instance['image']:
            t_i = tensorTr(i)
            if t_i.shape[0] == 1:
                t_i = torch.cat((t_i, t_i, t_i), dim=0)
            t_i = bb_transform(t_i)
            l_tensor.append(t_i.unsqueeze(0))
        t_img = torch.vstack(l_tensor).to(cfg.device)

        # INFERENCE
        z = bb_latent(t_img).view(t_img.size(0), -1)
        logit = bb_model(t_img).view(t_img.size(0), -1)
        return {
            'sim score': clip_outputs.logits_per_image.cpu(),
            'logits': logit.cpu(),
            'latent': z.cpu(),
        }

    # ITERATE OVER SPLIT
    for split in ['train', TEST_SPLIT[cfg.dataset]]:
        os.makedirs(os.path.join(path_to_save, split), exist_ok=True)
        logger.info(f'Generating {split} dataset {cfg.dataset}')
        # LOAD DATASET
        dataset = load_from_disk(os.path.join(cfg.dataset_dir,
                                              cfg.dataset))[split]
        dataset = dataset.shuffle()
        test = dataset.map(sim_score,
                           batched=True,
                           batch_size=cfg.batch_size,
                           remove_columns=["image"])
        save_file = os.path.join(path_to_save, split)
        test.save_to_disk(os.path.join(path_to_save, split))
        logger.info(f'Saved {split} dataset at {save_file}')


if __name__ == '__main__':
    generate_data()
