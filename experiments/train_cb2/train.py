import hashlib
import logging
import os
import pickle

import hydra
import torch
from cb2.dataset.constant import CLASSES, DIM_EMBEDDINGS, TEST_SPLIT
from cb2.dataset.loader import GENERATED_HF_CB2, load_concepts, load_similarity
from cb2.models.module import InterpreterCB2_HCI_CLS, InterpreterCB2_HCI_SEGMENT
from cb2.trainer.module import Trainer_CB2
from torch.utils.data import DataLoader


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
    
    # DEVICE
    device = cfg.device
    if device == "mcuda":                                                                                                                                                                                                                                                                                                                                                
        device = f"cuda:{int(os.environ['SLURM_ARRAY_TASK_ID'])%8}"
    logger.info(f"device: {device}")

    # LOAD CONCEPT DICO
    logger.info("Loading concept dictionnary...")
    concepts = load_concepts(concepts_bank_dir=cfg.path_dataset_dico, concepts_dico=cfg.concepts) 

    # DATASET DIRECTORY
    dataset_dir = os.path.join(
            cfg.path_dataset,
            cfg.dataset,
            cfg.clip_model,
            hashlib.sha256(str.encode(''.join(concepts))).hexdigest(), cfg.black_box
            )
    logger.info(f'Loading data from {dataset_dir}')
    dataset_train = GENERATED_HF_CB2(data_dir=dataset_dir, split='train')
    dataset_test = GENERATED_HF_CB2(data_dir=dataset_dir, split=TEST_SPLIT[cfg.dataset])
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=cfg.batch_size,
                                  drop_last=True,
                                  shuffle=True)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=cfg.batch_size,
                                 drop_last=True,
                                 shuffle=True)
    
    # INTERPRETER TYPES
    if  not cfg.concepts_per_head:
        concepts_per_head=len(concepts)
        interpreter = InterpreterCB2_HCI_CLS
    else:
        concepts_per_head=cfg.concepts_per_head
        interpreter = InterpreterCB2_HCI_SEGMENT

    # LOAD CB2 MODEL
    logger.info('Loading CB2 modules ...')
    interpret_model = interpreter(
        dim_embedding_space=DIM_EMBEDDINGS[cfg.clip_model],
        dim_conceptual_space=len(concepts),
        dim_white=concepts_per_head,
        dim_logits=CLASSES[cfg.dataset],
        whites_by_aggregator=cfg.whites_by_aggregator,
        type_of_hci=cfg.type_of_hci,
        dim_teacher_latent=DIM_EMBEDDINGS[cfg.black_box])

    # LOAD TRAINER
    trainer_model = Trainer_CB2
    trainer = trainer_model(device=device,
                            model=interpret_model,
                            temp=cfg.temp,
                            lr=cfg.lr,
                            r_ae=cfg.r_ae,
                            r_pred=cfg.r_pred,
                            r_align=cfg.r_align,
                            training_ablation=cfg.training_ablation,
                            )
    logger.info("Launching the training...")
    trainer.fit(data=dataloader_train,
                test_data=dataloader_test,
                epochs=cfg.epochs)
    torch.save(interpret_model,
               os.path.join(trainer.save_path, "interpreter.pk"))
    torch.save(
        interpret_model.state_dict(),
        os.path.join(trainer.save_path, "interpreter_state.pk"),
    )
    logger.info(f"Model saved at {trainer.save_path}")


if __name__=='__main__':
    train()
