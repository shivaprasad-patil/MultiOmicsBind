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


def train_temporal_model(
    dataset,
    device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    binding_modality: str = 'transcriptomics',
    embed_dim: int = 256,
    dropout: float = 0.2,
    temporal_encoder_kwargs: Optional[Dict[str, Any]] = None,
    contrastive_weight: float = 0.1,
    val_split: float = 0.2,
    verbose: bool = True
):
    """
    Train a temporal multi-omics model with standard configuration.
    
    This function provides a high-level interface for training temporal multi-omics
    models with minimal boilerplate code. It handles model initialization, data
    splitting, training loop, and history tracking.
    
    Args:
        dataset: TemporalMultiOmicsDataset instance
        device: torch device ('cuda' or 'cpu')
        epochs (int): Number of training epochs (default: 20)
        batch_size (int): Batch size for training (default: 32)
        lr (float): Learning rate (default: 1e-4)
        binding_modality (str): Modality to use for binding/alignment (default: 'transcriptomics')
        embed_dim (int): Embedding dimension (default: 256)
        dropout (float): Dropout rate (default: 0.2)
        temporal_encoder_kwargs (Optional[Dict]): Dict of temporal encoder configurations
        contrastive_weight (float): Weight for contrastive loss (default: 0.1)
        val_split (float): Validation split ratio (default: 0.2)
        verbose (bool): Whether to print training progress (default: True)
    
    Returns:
        model: Trained TemporalMultiOmicsBind model
        history: Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_acc': List of training accuracies per epoch
            - 'val_acc': List of validation accuracies per epoch
    
    Example:
        >>> from multiomicsbind import TemporalMultiOmicsDataset, train_temporal_model
        >>> dataset = TemporalMultiOmicsDataset(...)
        >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
        >>> model, history = train_temporal_model(dataset, device, epochs=15)
        >>> print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    """
    from torch.utils.data import random_split
    from ..core.model import TemporalMultiOmicsBind
    
    if temporal_encoder_kwargs is None:
        temporal_encoder_kwargs = {}
    
    # Handle both raw datasets and Subset objects from random_split
    # If dataset is a Subset, get the original dataset
    if hasattr(dataset, 'dataset'):
        # This is a Subset from random_split
        base_dataset = dataset.dataset
    else:
        # This is the original dataset
        base_dataset = dataset
    
    # Split dataset if val_split > 0
    if val_split > 0:
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        # No additional split needed (user already split externally)
        train_dataset = dataset
        val_dataset = None
        train_size = len(dataset)
        val_size = 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    # Get number of classes from all labels (use base_dataset)
    all_labels = [base_dataset[i]['label'] for i in range(len(base_dataset))]
    num_classes = len(np.unique(all_labels))
    
    if verbose:
        print(f"Training with {num_classes} classes")
        print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize model (use base_dataset methods)
    static_dims = {k: v for k, v in base_dataset.get_input_dims().items() if k in base_dataset.static_data}
    temporal_dims = {k: v for k, v in base_dataset.get_input_dims().items() if k in base_dataset.temporal_data}
    temporal_encoders = {k: 'lstm' for k in temporal_dims}
    cat_dims, num_dims = base_dataset.get_metadata_dims()
    
    model = TemporalMultiOmicsBind(
        static_input_dims=static_dims,
        temporal_input_dims=temporal_dims,
        temporal_encoders=temporal_encoders,
        binding_modality=binding_modality,
        cat_dims=cat_dims,
        num_dims=num_dims,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=dropout,
        temporal_encoder_kwargs=temporal_encoder_kwargs
    ).to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cls_criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") if verbose else train_loader
        for batch in train_iter:
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            labels = inputs.pop('label').to(device)
            
            optimizer.zero_grad()
            logits, embeddings = model(inputs, return_embeddings=True)
            
            # Combined loss: classification + contrastive
            cls_loss = cls_criterion(logits, labels)
            contrast_loss = contrastive_loss(embeddings, binding_modality=binding_modality)
            loss = cls_loss + contrastive_weight * contrast_loss
            
            loss.backward()
            
            # Add gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
            if verbose and isinstance(train_iter, tqdm):
                train_iter.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        if val_loader is not None:
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]") if verbose else val_loader
                for batch in val_iter:
                    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                             for k, v in batch.items()}
                    labels = inputs.pop('label').to(device)
                    
                    logits, embeddings = model(inputs, return_embeddings=True)
                    cls_loss = cls_criterion(logits, labels)
                    contrast_loss = contrastive_loss(embeddings, binding_modality=binding_modality)
                    loss = cls_loss + contrastive_weight * contrast_loss
                    
                    val_loss += loss.item()
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        
        if val_loader is not None:
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_correct / val_total)
        else:
            history['val_loss'].append(0.0)  # No validation
            history['val_acc'].append(0.0)
        
        if verbose:
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Train Acc: {history['train_acc'][-1]:.4f}, "
                      f"Val Loss: {history['val_loss'][-1]:.4f}, "
                      f"Val Acc: {history['val_acc'][-1]:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Train Acc: {history['train_acc'][-1]:.4f}")
    
    return model, history
