import torch
import torch.nn as nn
from cb2.trainer.abstract import Trainer


class Trainer_BB(Trainer):

    def __init__(
            self,
            model,
            r_pred,
            transform,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            training_ablation=1
            ):
        super().__init__(model, device, lr, training_ablation)
        self.r_pred = r_pred
        self.transform=transform
    
    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_accuracy = "accuracy"
        keys = [
                self.key_loss_global,
                self.key_accuracy
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_loss_global = 0  # global
        # RUNNING ACC
        self.running_accuracy = 0
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_accuracy].append(self.running_accuracy)

    def global_loss(self, forward_pass, y):
        l_ce = nn.CrossEntropyLoss()
        # PREDICTION
        l_pred = self.r_pred * l_ce(forward_pass, y)
        # GLOBAL LOSS
        loss = l_pred
        # ACCURACY
        y_f = torch.argmax(torch.nn.functional.softmax(forward_pass.detach(),dim=1),dim=1)
        accuracy = (y_f == y).sum().item()
        return {
                'loss':loss,
                'accuracy': accuracy,
                }
    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(x)
            ## LOSSES
            losses = self.global_loss(forward_pass, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_loss_global += losses['loss'].item()/n
            # ACCURACY HISTORY
            self.running_accuracy += losses['accuracy']/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
             if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f} acc :{self.running_accuracy*n/(data.batch_size*(i+1)):.4f}"
                )   
        self.write_history(train=train)
      

class Trainer_CB2(Trainer):
    
    def __init__(
            self,
            model,
            r_ae ,
            r_pred,
            r_align,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=1e-3,
            temp=1,
            training_ablation=1
            ):
        super().__init__(model, device, lr, training_ablation)
        # ATTRIBUTES
        self.r_ae = r_ae
        self.r_pred = r_pred
        self.r_align = r_align
        self.temp = temp

    def history_init(self):
        self.history = {}
        self.history_test = {}
        self.key_loss_global = "global loss"
        self.key_loss_ae = "reconstruction loss"
        self.key_loss_pred = "prediction loss (on logits)"
        self.key_loss_align = "alignement loss (on latent)"
        self.key_faithfulness = "faithfulnes"
        self.key_accuracy = "accuracy"
        keys = [
                self.key_loss_global,
                self.key_loss_ae,
                self.key_loss_pred,
                self.key_loss_align,
                self.key_faithfulness,
                self.key_accuracy
                ]
        for key in keys:
            self.history[key] = []
            self.history_test[key] = []

    def  init_running_losses(self):
        # RUNNING LOSSES
        self.running_loss_global = 0  # global
        self.running_loss_ae = 0
        self.running_loss_pred = 0
        self.running_loss_align = 0
        # RUNNING ACC
        self.running_accuracy = 0
        self.running_faithfulness = 0
 
    def write_history(self, train=True):
        histo = self.history if train else self.history_test
        histo[self.key_loss_global].append(self.running_loss_global)    
        histo[self.key_loss_ae].append(self.running_loss_ae)    
        histo[self.key_loss_align].append(self.running_loss_align)    
        histo[self.key_loss_pred].append(self.running_loss_pred)
        histo[self.key_accuracy].append(self.running_accuracy)
        histo[self.key_faithfulness].append(self.running_faithfulness)

    def global_loss(self, forward_pass, z, logits, y):
        l_2 = nn.MSELoss()
        s = torch.nn.Softmax(dim=1)
        l_ce = nn.CrossEntropyLoss()
        # ALIGNEMENT (LATENT)
        p_f = s(z / self.temp)
        l_align = (self.r_align * l_ce(forward_pass["z_tilda"] / self.temp, p_f) /self.temp)
        # RECONSTRUCTION
        l_ae = self.r_ae * l_2(forward_pass["z_bar"], forward_pass["z"])
        # PREDICTION
        l_pred = self.r_pred * l_2(forward_pass["logits"], logits)
        # GLOBAL LOSS
        loss = l_ae +  l_align + l_pred
        # ACCURACY
        y_f = torch.argmax(torch.nn.functional.softmax(logits.detach(),dim=1),dim=1)
        y_cb2 = torch.argmax(torch.nn.functional.softmax(forward_pass['logits'].detach(),dim=1),dim=1)
        accuracy = (y_cb2 == y).sum().item()
        faithfulnes = (y_cb2 == y_f).sum().item()
        return {
                'loss':loss,
                'loss_ae': l_ae,
                'loss_pred': l_pred,
                'loss_align': l_align,
                'accuracy': accuracy,
                'faithfulnes': faithfulnes
                }


    def epoch(self, data, train=True):
        self.model.train() if train else self.model.eval()
        # LOSS DEFINITION
        n_batch = len(data)
        n = n_batch * data.batch_size
        for i, batch in enumerate(data):
            # TRAINING SAMPLES ABLATION
            if i / n_batch > self.training_ablation:
                self.logger.info(f"breaking at batch {i}/{n_batch}")
                break
            # GET DATA
            z, psi, logits, y, c = batch
            z = z.to(self.device)
            psi = torch.softmax(psi, dim=1).to(self.device)
            logits = logits.to(self.device)
            y = y.to(self.device)
            c = c.to(self.device)
            # FORWARD ON INTERPRETER
            forward_pass = self.model(z, psi, c, device=self.device)
            ## LOSSES
            losses = self.global_loss(forward_pass, z, logits, y)
            if train:
                # BACKWARD PASS
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            # LOSS HISTORY
            self.running_loss_global += losses['loss'].item()/n
            self.running_loss_pred += losses['loss_pred'].item()/n
            self.running_loss_ae += losses['loss_ae'].item()/n
            self.running_loss_align += losses['loss_align'].item()/n
            # ACCURACY HISTORY
            self.running_accuracy += losses['accuracy']/n
            self.running_faithfulness += losses['faithfulnes']/n
            # VERBOSE
            if i % (n_batch // 5 + 1) == n_batch // 5 - 1:
                self.logger.info(
                        f"|\tbatch {i}/{n_batch}, total loss :{self.running_loss_global/n_batch:.4f} acc :{self.running_accuracy*n/(data.batch_size*(i+1)):.4f}"
                )   
        self.write_history(train=train)
