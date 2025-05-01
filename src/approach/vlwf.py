import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

class ELBO(nn.Module):
    def __init__(self, kl_weight=1.0):
        """
        Initialize the ELBO loss class.

        Args:
            kl_weight (float): Weight for the KL divergence term.
        """
        super(ELBO, self).__init__()
        self.kl_weight = kl_weight

    def forward(self, logits, aux_targets, kl_divergence):
        """
        Compute the ELBO loss.

        Args:
            logits (torch.Tensor): Logits output from the model.
            aux_targets (torch.Tensor): Auxiliary targets for the NLL loss.
            kl_divergence (torch.Tensor): KL divergence term from the model.

        Returns:
            torch.Tensor: The computed ELBO loss.
        """
        # Compute the negative log-likelihood loss
        nll_loss = F.nll_loss(F.log_softmax(logits, dim=1), aux_targets)

        # Combine NLL loss and KL divergence
        elbo_loss = nll_loss + self.kl_weight * kl_divergence

        return elbo_loss
    


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, 
                 model, 
                 device, 
                 nepochs=100, 
                 lr=0.05, 
                 lr_min=1e-4, 
                 lr_factor=3, 
                 lr_patience=5, 
                 clipgrad=10000,
                 momentum=0, 
                 wd=0, multi_softmax=False, 
                 wu_nepochs=0, 
                 wu_lr_factor=1, 
                 fix_bn=False, 
                 eval_on_train=False,
                 logger=None, 
                 exemplars_dataset=None, 
                 lamb=1, 
                 T=2
        ):
        super(Appr, self).__init__(model, 
                                   device, 
                                   nepochs, 
                                   lr, lr_min, 
                                   lr_factor, 
                                   lr_patience, 
                                   clipgrad, 
                                   momentum, 
                                   wd,
                                   multi_softmax, 
                                   wu_nepochs, 
                                   wu_lr_factor, 
                                   fix_bn, 
                                   eval_on_train, 
                                   logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.kl_weight = 1e-5
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
       
        parser.add_argument('--lamb', default=1, type=float, required=False,help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--T', default=2, type=int, required=False,help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        
        for images, targets in trn_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs_old = None
            features_old = None
            if t > 0:
                outputs_old, features_old = self.model_old(images, return_features=True)
            outputs, features = self.model(images, return_features=True)
            # loss = self.criterion(t, outputs, targets, outputs_old) 
            # loss = self.rcriterion(t, outputs, targets, outputs_old, features, features_old)
            # loss = self.focal_loss(outputs[t], targets - self.model.task_offset[t])
            loss = self.ELBO(t, outputs, targets, outputs_old, features, features_old)
            # loss  = self.focal_loss(t, outputs, targets)
            kl = self.model.KL()
            loss = loss + self.kl_weight * kl
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        
    def eval_proto(self, t, val_loader):
        """Evaluation using Mahalanobis Distance (task-aware and task-agnostic)"""
        with torch.no_grad():
            self.model.eval()
            features_by_class = {}
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0

            # First pass: collect features for prototypes
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                _, features = self.model(images, return_features=True)
                for feat, label in zip(features, targets):
                    label = label.item()
                    if label not in features_by_class:
                        features_by_class[label] = []
                    features_by_class[label].append(feat)

            # Compute class prototypes and shared covariance
            all_features = []
            prototypes = {}
            for cls, feats in features_by_class.items():
                feats_tensor = torch.stack(feats)
                proto = feats_tensor.mean(dim=0)
                prototypes[cls] = proto
                all_features.append(feats_tensor)

            all_features = torch.cat(all_features)
            centered = all_features - all_features.mean(dim=0)
            cov = centered.T @ centered / (len(all_features) - 1)
            cov += 1e-6 * torch.eye(cov.size(0), device=self.device)
            inv_cov = torch.linalg.inv(cov)

            # Class index mapping
            all_classes = sorted(prototypes.keys())
            class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

            # Second pass: evaluate distances
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))

                _, features = self.model(images, return_features=True)

                # Compute Mahalanobis distances to all prototypes
                dists = []
                for cls in all_classes:
                    proto = prototypes[cls]
                    diff = features - proto
                    dist = torch.sqrt((diff @ inv_cov * diff).sum(dim=1))
                    dists.append(dist.unsqueeze(1))

                dists = torch.cat(dists, dim=1)  # [batch_size, num_classes]
                log_probs = F.log_softmax(-dists, dim=1)  # negative because closer = better

                # Map target labels to indices in prototypes
                mapped_targets = torch.tensor([class_to_idx[y.item()] for y in targets], device=self.device)

                # Loss (just basic NLL here, no extra args)
                loss = F.nll_loss(log_probs, mapped_targets)

                # Prediction
                preds = log_probs.argmax(dim=1)

                # Task-Aware accuracy
                classes_per_task = self.model.task_cls[t]
                raw_task_class_indices = list(range(t * classes_per_task, (t + 1) * classes_per_task))
                task_class_indices = [class_to_idx[c] for c in raw_task_class_indices if c in class_to_idx]

                task_dists = dists[:, task_class_indices]
                task_preds = task_dists.argmin(dim=1)
                task_true = torch.tensor([
                    task_class_indices.index(class_to_idx[y.item()])
                    for y in targets
                ], device=self.device)

                hits_taw = (task_preds == task_true).float()
                hits_tag = (preds == mapped_targets).float()

                # Logging
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

            return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num


    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
  
        
        
    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    

    def rcriterion(self, t, outputs, targets, outputs_old=None, features=None, features_old=None):
        """Returns the loss value with both logit and feature distillation.
        
        Args:
            t (int): Current task index
            outputs (list of tensors): Outputs from all heads of the current model
            targets (tensor): Ground truth labels
            outputs_old (list of tensors, optional): Outputs from all heads of the old model (for logit distillation)
            features (tensor, optional): Features from the current model (before the heads)
            features_old (tensor, optional): Features from the old model (for feature distillation)
        """
        loss = 0
        
        # Hyperparameters for balancing the losses
        lamb_logit = self.lamb  # Weight for logit distillation (from your original code)
        lamb_feature = 1.0      # Weight for feature distillation (tune this as needed)
        T = self.T              # Temperature for logit distillation
        
        if t > 0:
            # Logit distillation loss for all previous tasks
            loss += lamb_logit * self.cross_entropy(
                torch.cat(outputs[:t], dim=1), 
                torch.cat(outputs_old[:t], dim=1), 
                exp=1.0 / T
            )
            
            # Feature distillation loss (e.g., MSE between current and old features)
            if features is not None and features_old is not None:
                loss += lamb_feature * torch.nn.functional.mse_loss(features, features_old)
        
        # Current task cross-entropy loss
        if len(self.exemplars_dataset) > 0:
            # With exemplars, use all heads
            loss += torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            # Without exemplars, use only the current task head
            loss += torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        
        return loss
    

    def ELBO(self, t, outputs, targets, outputs_old=None, features=None, features_old=None):
        """Returns the loss value with both logit and feature distillation.
        
        Args:
            t (int): Current task index
            outputs (list of tensors): Outputs from all heads of the current model
            targets (tensor): Ground truth labels
            outputs_old (list of tensors, optional): Outputs from all heads of the old model (for logit distillation)
            features (tensor, optional): Features from the current model (before the heads)
            features_old (tensor, optional): Features from the old model (for feature distillation)
        """
        loss = 0
        
        # Hyperparameters for balancing the losses
        lamb_logit = self.lamb  # Weight for logit distillation (from your original code)
        lamb_feature = 1.0      # Weight for feature distillation (tune this as needed)
        T = self.T              # Temperature for logit distillation
        
        if t > 0:
            # Logit distillation loss for all previous tasks
            loss += lamb_logit * self.cross_entropy(
                torch.cat(outputs[:t], dim=1), 
                torch.cat(outputs_old[:t], dim=1), 
                exp=1.0 / T
            )
            
            # Feature distillation loss (e.g., MSE between current and old features)
            if features is not None and features_old is not None:
                loss += lamb_feature * torch.nn.functional.mse_loss(features, features_old)
        
        # Current task NLL loss
        if len(self.exemplars_dataset) > 0:
            # With exemplars, use all heads
            loss += torch.nn.functional.nll_loss(
                torch.log_softmax(torch.cat(outputs, dim=1), dim=1), targets
            )
        else:
            # Without exemplars, use only the current task head
            loss += torch.nn.functional.nll_loss(
                torch.log_softmax(outputs[t], dim=1), targets - self.model.task_offset[t]
            )
        
        return loss
    

    def focal_loss(self, t, outputs, targets, alpha=0.25, gamma=2.0):
        """Focal loss implementation"""
        BCE_loss = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        return F_loss
    

    
