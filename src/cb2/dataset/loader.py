import hashlib
import logging
import os
import pickle

import hydra
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import torchvision.models as m
from transformers import CLIPModel, CLIPProcessor


class GENERATED_HF_CB2(Dataset):

    def __init__(self,
                data_dir,
                split,
                transform=None):
        # LOGGER
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Loading {split} dataset ...')
        # ATTRIBUTES
        self.transform = transform
        dataset = load_from_disk(os.path.join(data_dir, split))
        self.dataset = dataset.with_format('torch')
    
    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, idx):
        d = self.dataset[idx]
        if 'conceptBase' in d.keys():
            return d['latent'], d['sim score'], d['logits'], d['label'], d['conceptBase']
        if 'fine_label' in d.keys():
            return d['latent'], d['sim score'], d['logits'], d['fine_label'], torch.ones(2,2)
        return d['latent'], d['sim score'], d['logits'], d['label'], torch.ones(2,2)


class HF_DATASET_TORCH(Dataset):

    def __init__(self,
                dataset,
                split,
                transform,
                data_dir=None,
                ):
        # LOGGER
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Loading {split} dataset ...')
        # ATTRIBUTES
        self.transform = transform
        if data_dir:
            self.logger.info(f'Loading {split} data from {data_dir}')
            data = load_from_disk(os.path.join(data_dir, dataset))[split]
        else:
            self.logger.info(f'Loading {split} dataset {dataset} from huggingface hub')
            data = load_dataset(dataset)[split]
        self.dataset = data.with_format('torch')
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        if batch['image'].dim()==2:
            temp = batch['image'].unsqueeze(0)
            x = torch.cat((temp, temp, temp), dim=0)
        elif batch['image'].dim()==3:
            x = batch['image'].permute(2,0,1)
        x = self.transform(x)
        y = batch['label']
        return x, y 


class HUGGING_FACE_CB2(Dataset):

    def __init__(self,
                 dataset,
                 clip_model='openai/clip-vit-large-patch14',
                 concept_list=['dog'],
                 split='train',
                 transform=None):
        # LOGGER
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Loading {split} dataset ...')
        # ATTRIBUTES
        self.transform = transform
        self.concept_list = concept_list
        # LOAD CLIP MODEL
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        # LOAD IMAGE DATAT
        dataset = load_dataset(dataset, split=split, streaming=True)
        self.dataset = dataset.with_format('torch')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['image'].permute(2, 0, 1)  # HWC -> CHW
        label = data['label']
        inputs = self.processor(text=self.concept_list,
                                images=img,
                                return_tensors='pt',
                                padding=True)
        clip_outputs = self.clip_model(**inputs)
        if self.transform:
            img = self.transform(img)
        return img, clip_outputs.logits_per_image.flatten(), label

def load_concepts(concepts_bank_dir, concepts_dico):
    with open(os.path.join(concepts_bank_dir, concepts_dico), 'rb') as handle:
        l = pickle.load(handle)
    l = [s.replace('_', ' ') for s in l]
    return l

def load_similarity(sim_dir, sim_source, concepts):
    if sim_source == "none":
        n = len(concepts)
        return torch.ones(n,n)
    return torch.from_numpy(np.load(
        os.path.join(sim_dir,sim_source,hashlib.sha256(str.encode(''.join(concepts))).hexdigest()+ ".npy")
        ))

def load_black_box(dataset, black_box, device, generated_model_dir=''):
    if black_box=='resnet18':
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'{black_box}.pk'), map_location=device)
        bb_transform = ResNet18_Weights.DEFAULT.transforms()
    elif black_box=="vitB16":
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        bb_transform = m.ViT_B_16_Weights.DEFAULT.transforms() 
    elif black_box=='vgg16':
        model = torch.load(os.path.join(generated_model_dir, dataset, black_box, f'resnet18.pk'), map_location=device)
        bb_transform = m.VGG16_Weights.DEFAULT.transforms()
    else:
        return
    return model, bb_transform

