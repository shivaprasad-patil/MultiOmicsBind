"""
Training utilities and trainer class for MultiOmicsBind.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
import numpy as np

from ..core.losses import contrastive_loss


def train_multiomicsbind(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 20,
    temperature: float = 0.07,
    use_classification: bool = False,
    contrastive_weight: float = 1.0,
    classification_weight: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Train MultiOmicsBind model with contrastive and/or classification objectives.
    
    Args:
        model (nn.Module): MultiOmicsBind model to train
        dataloader (DataLoader): Training data loader
        optimizer (optim.Optimizer): Optimizer for training
        device (torch.device): Device to train on
        epochs (int): Number of training epochs (default: 20)
        temperature (float): Temperature for contrastive loss (default: 0.07)
        use_classification (bool): Whether to use classification loss (default: False)
        contrastive_weight (float): Weight for contrastive loss (default: 1.0)
        classification_weight (float): Weight for classification loss (default: 1.0)
        scheduler (Optional): Learning rate scheduler
        verbose (bool): Whether to print training progress (default: True)
        
    Returns:
        nn.Module: Trained model
        
    Example:
        >>> model = MultiOmicsBindWithHead(input_dims, num_classes=3)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> trained_model = train_multiomicsbind(
        ...     model, dataloader, optimizer, device,
        ...     epochs=50, use_classification=True
        ... )
    """
    model.train()
    criterion_clf = nn.CrossEntropyLoss()
    
    # Track training metrics
    epoch_losses = []
    epoch_contrastive_losses = []
    epoch_classification_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_clf_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0
        
        # Progress bar
        if verbose:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = dataloader

        for batch in pbar:
            # Move batch to device
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    inputs[k] = {k2: v2.to(device) for k2, v2 in v.items()}
                elif isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v

            optimizer.zero_grad()
            
            # Forward pass
            if use_classification and "label" in inputs:
                # Classification mode: get both logits and embeddings
                logits, embeddings = model(inputs, return_embeddings=True)
                
                # Classification loss
                loss_clf = criterion_clf(logits, inputs["label"])
                
                # Contrastive loss (if multiple modalities)
                if len(embeddings) > 1:
                    loss_con = model.compute_contrastive_loss(embeddings, temperature)
                    total_loss_batch = (classification_weight * loss_clf + 
                                      contrastive_weight * loss_con)
                else:
                    loss_con = torch.tensor(0.0, device=device)
                    total_loss_batch = classification_weight * loss_clf
                    
            else:
                # Contrastive-only mode
                embeddings = model(inputs)
                loss_con = model.compute_contrastive_loss(embeddings, temperature)
                loss_clf = torch.tensor(0.0, device=device)
                total_loss_batch = contrastive_weight * loss_con

            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping (optional, helps with stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Update metrics
            total_loss += total_loss_batch.item()
            total_clf_loss += loss_clf.item()
            total_contrastive_loss += loss_con.item()
            num_batches += 1
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'Loss': f'{total_loss_batch.item():.4f}',
                    'Con': f'{loss_con.item():.4f}',
                    'Clf': f'{loss_clf.item():.4f}'
                })

        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_clf = total_clf_loss / num_batches
        avg_con = total_contrastive_loss / num_batches
        
        # Store metrics
        epoch_losses.append(avg_loss)
        epoch_contrastive_losses.append(avg_con)
        epoch_classification_losses.append(avg_clf)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {avg_loss:.4f}, "
                  f"Contrastive: {avg_con:.4f}, "
                  f"Classification: {avg_clf:.4f}, "
                  f"LR: {current_lr:.2e}")

    if verbose:
        print("Training complete!")
    
    # Store training history in model
    model.training_history = {
        'total_loss': epoch_losses,
        'contrastive_loss': epoch_contrastive_losses,
        'classification_loss': epoch_classification_losses
    }
    
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_classification: bool = False
) -> Dict[str, float]:
    """
    Evaluate trained model on validation/test data.
    
    Args:
        model (nn.Module): Trained model to evaluate
        dataloader (DataLoader): Evaluation data loader
        device (torch.device): Device to evaluate on
        use_classification (bool): Whether to compute classification metrics
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    model.eval()
    
    total_samples = 0
    total_contrastive_loss = 0.0
    total_classification_loss = 0.0
    correct_predictions = 0
    
    criterion_clf = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    inputs[k] = {k2: v2.to(device) for k2, v2 in v.items()}
                elif isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v
            
            batch_size = len(next(iter(inputs.values())))
            total_samples += batch_size
            
            if use_classification and "label" in inputs:
                logits, embeddings = model(inputs, return_embeddings=True)
                
                # Classification metrics
                loss_clf = criterion_clf(logits, inputs["label"])
                total_classification_loss += loss_clf.item() * batch_size
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == inputs["label"]).sum().item()
                
                # Contrastive loss
                if len(embeddings) > 1:
                    loss_con = model.compute_contrastive_loss(embeddings)
                    total_contrastive_loss += loss_con.item() * batch_size
            else:
                embeddings = model(inputs)
                if len(embeddings) > 1:
                    loss_con = model.compute_contrastive_loss(embeddings)
                    total_contrastive_loss += loss_con.item() * batch_size
    
    # Calculate metrics
    metrics = {}
    
    if total_contrastive_loss > 0:
        metrics['contrastive_loss'] = total_contrastive_loss / total_samples
    
    if use_classification and total_classification_loss > 0:
        metrics['classification_loss'] = total_classification_loss / total_samples
        metrics['accuracy'] = correct_predictions / total_samples
    
    return metrics


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Current model
            
        Returns:
            bool: True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()
