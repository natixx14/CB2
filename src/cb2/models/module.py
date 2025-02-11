from dataclasses import dataclass

import numpy as np
import torch
from cb2.neurhci.neurhci_torch import HCI_2_add_2_layers_partition, HCI_2_add_balanced
from torch import nn


@dataclass(unsafe_hash=True)
class InterpreterCB2_HCI_CLS(nn.Module):
    dim_white: int = 50
    dim_teacher_latent: int = 64
    dim_embedding_space: int = 768
    dim_conceptual_space: int = 2096
    dim_logits: int = 10
    whites_by_aggregator: int = 5
    type_of_hci: str = "balanced"

    def __post_init__(self):
        super().__init__()

        # ALIGNEMENT MODULE
        self.entangler = nn.Sequential(
            nn.Linear(int(self.dim_white/self.whites_by_aggregator)*self.dim_logits, 100),
            nn.SiLU(),
            nn.Linear(100, self.dim_teacher_latent),
        )
        self.disentangler = nn.Sequential(
            nn.Linear(self.dim_teacher_latent, 100),
            nn.SiLU(),
            nn.Linear(100, int(self.dim_white/self.whites_by_aggregator)*self.dim_logits),
        )

        # NORMALISATION
        self.batch_norm = nn.BatchNorm1d(self.dim_white, affine=False)
        
        # HCI DEFINITION
        self.hci_node_types = {i: "ID" for i in range(self.dim_white)}
        if self.type_of_hci == "balanced":
            self.heads = nn.ModuleList([
                HCI_2_add_balanced(self.whites_by_aggregator, self.dim_white)
                for _ in range(self.dim_logits)
            ])
        elif self.type_of_hci == "2_layers":
            self.heads = nn.ModuleList([
                HCI_2_add_2_layers_partition(self.whites_by_aggregator,
                                             self.dim_white)
                for _ in range(self.dim_logits)
            ])
        else:
            raise (ValueError(
                "Not a valid HCI type, must be 'balanced' or '2_layers'"))

    def forward(self, z, psi, c, device="cpu"):
        batch_size = psi.shape[0]
        logits = torch.zeros((batch_size, self.dim_logits, 1))
        snz_list = []
        for i, l in enumerate(self.heads):
            p = self.batch_norm(psi)
            dico = l(psi)
            logits[:, i] = dico['output'].unsqueeze(-1)  
            snz_list.append(dico['snz'])
        snz = torch.hstack(snz_list)
        z_tilda = self.entangler(snz)
        z_bar = self.disentangler(z_tilda)
        return {
            "z": snz,
            "z_tilda": z_tilda,
            "z_bar": z_bar,
            "snz": snz,
            "logits": torch.squeeze(logits, dim=2).to(device),
        }



@dataclass(unsafe_hash=True)
class InterpreterCB2_HCI_SEGMENT(nn.Module):
    dim_white: int = 50
    dim_teacher_latent: int = 64
    dim_embedding_space: int = 768
    dim_conceptual_space: int = 2096
    dim_logits: int = 10
    whites_by_aggregator: int = 5
    type_of_hci: str = "balanced"

    def __post_init__(self):
        super().__init__()

        self.hci_node_types = {i: "ID" for i in range(self.dim_white)}
        self.entangler = nn.Sequential(
            nn.Linear(int(self.dim_white/self.whites_by_aggregator)*self.dim_logits, 100),
            nn.SiLU(),
            nn.Linear(100, self.dim_teacher_latent),
        )
        self.disentangler = nn.Sequential(
            nn.Linear(self.dim_teacher_latent, 100),
            nn.SiLU(),
            nn.Linear(100, int(self.dim_white/self.whites_by_aggregator)*self.dim_logits),
        )
        self.batch_norm = nn.BatchNorm1d(self.dim_white, affine=False)
        if self.type_of_hci == "balanced":
            self.heads = nn.ModuleList([  #HERE
                HCI_2_add_balanced(self.whites_by_aggregator, self.dim_white)
                for _ in range(self.dim_logits)
            ])
        elif self.type_of_hci == "2_layers":
            self.heads = nn.ModuleList([  #HERE
                HCI_2_add_2_layers_partition(self.whites_by_aggregator,
                                             self.dim_white)
                for _ in range(self.dim_logits)
            ])
        else:
            raise (ValueError(
                "Not a valid HCI type, must be 'balanced' or '2_layers'"))

    def forward(self, z, psi, c, device="cpu"):
        batch_size = psi.shape[0]
        logits = torch.zeros((batch_size, self.dim_logits, 1))
        snz_list = []
        for i, l in enumerate(self.heads):
            pp = psi[:,i*self.dim_white:(i+1)*self.dim_white]
            p = torch.nn.functional.softmax(pp, dim=1)
            p = self.batch_norm(p)
            dico = l(p)
            logits[:, i] = dico['output'].unsqueeze(-1)  
            snz_list.append(dico['snz'])
        snz = torch.hstack(snz_list)
        z_tilda = self.entangler(snz)
        z_bar = self.disentangler(z_tilda)
        return {
            "z": snz,
            "z_tilda": z_tilda,
            "z_bar": z_bar,
            "snz": snz,
            "logits": torch.squeeze(logits, dim=2).to(device),          
            }






