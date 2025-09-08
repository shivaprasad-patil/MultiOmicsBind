"""
Dataset classes for multi-omics data loading and preprocessing.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import warnings


class MultiOmicsDataset(Dataset):
    """
    PyTorch Dataset for multi-omics data with metadata support.
    
    This dataset handles loading and preprocessing of multiple omics modalities
    along with optional metadata and labels. It automatically aligns samples
    across modalities and handles missing data.
    
    Args:
        data_paths (Dict[str, str]): Dictionary mapping modality names to file paths
        metadata_path (Optional[str]): Path to metadata CSV file
        cat_cols (Optional[List[str]]): List of categorical metadata columns
        num_cols (Optional[List[str]]): List of numerical metadata columns  
        label_col (Optional[str]): Name of label column for supervised learning
        sample_id_col (str): Name of sample ID column (default: "sample_id")
        normalize (bool): Whether to normalize omics data (default: True)
        
    Example:
        >>> data_paths = {
        ...     'transcriptomics': 'tx_data.csv',
        ...     'proteomics': 'pr_data.csv',
        ...     'cell_painting': 'cp_data.csv'
        ... }
        >>> dataset = MultiOmicsDataset(
        ...     data_paths=data_paths,
        ...     metadata_path='metadata.csv',
        ...     cat_cols=['drug', 'cell_line'],
        ...     num_cols=['dose'],
        ...     label_col='response'
        ... )
    """
    
    def __init__(
        self,
        data_paths: Dict[str, str],
        metadata_path: Optional[str] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        sample_id_col: str = "sample_id",
        normalize: bool = True
    ):
        self.data_paths = data_paths
        self.metadata_path = metadata_path
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.label_col = label_col
        self.sample_id_col = sample_id_col
        self.normalize = normalize
        
        # Load and process data
        self._load_data()
        self._process_metadata()
        self._align_samples()
        
        print(f"Dataset initialized with {len(self)} samples and {len(self.modalities)} modalities")
        print(f"Modalities: {self.modalities}")
        if self.labels is not None:
            print(f"Labels: {len(np.unique(self.labels))} classes")

    def _load_data(self):
        """Load omics data from CSV files."""
        self.omics_data = {}
        self.feature_names = {}
        
        for modality, path in self.data_paths.items():
            try:
                df = pd.read_csv(path)
                if self.sample_id_col not in df.columns:
                    raise ValueError(f"Sample ID column '{self.sample_id_col}' not found in {path}")
                
                df = df.set_index(self.sample_id_col)
                self.omics_data[modality] = df
                self.feature_names[modality] = df.columns.tolist()
                
                print(f"Loaded {modality}: {df.shape[0]} samples, {df.shape[1]} features")
                
            except Exception as e:
                warnings.warn(f"Failed to load {modality} from {path}: {e}")
                continue
        
        self.modalities = list(self.omics_data.keys())
        
        if not self.modalities:
            raise ValueError("No omics data successfully loaded")

    def _process_metadata(self):
        """Load and process metadata."""
        self.metadata = None
        self.cat_data = None
        self.num_data = None
        self.labels = None
        self.cat_encoders = {}
        
        if self.metadata_path:
            try:
                self.metadata = pd.read_csv(self.metadata_path).set_index(self.sample_id_col)
                print(f"Loaded metadata: {self.metadata.shape[0]} samples")
                
                # Process categorical columns
                if self.cat_cols:
                    cat_data = []
                    for col in self.cat_cols:
                        if col in self.metadata.columns:
                            # Convert to categorical codes
                            codes, uniques = pd.factorize(self.metadata[col])
                            self.cat_encoders[col] = uniques
                            cat_data.append(codes)
                        else:
                            warnings.warn(f"Categorical column '{col}' not found in metadata")
                    
                    if cat_data:
                        self.cat_data = np.column_stack(cat_data)
                
                # Process numerical columns
                if self.num_cols:
                    num_cols_available = [col for col in self.num_cols if col in self.metadata.columns]
                    if num_cols_available:
                        self.num_data = self.metadata[num_cols_available].values.astype(np.float32)
                        if self.normalize:
                            # Standardize numerical features
                            self.num_data = (self.num_data - np.mean(self.num_data, axis=0)) / (np.std(self.num_data, axis=0) + 1e-8)
                
                # Process labels
                if self.label_col and self.label_col in self.metadata.columns:
                    label_codes, label_uniques = pd.factorize(self.metadata[self.label_col])
                    self.labels = label_codes
                    self.label_encoder = label_uniques
                    print(f"Labels: {len(label_uniques)} classes - {label_uniques.tolist()}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load metadata from {self.metadata_path}: {e}")

    def _align_samples(self):
        """Align samples across all modalities and metadata."""
        # Find common sample IDs
        sample_ids = set(self.omics_data[self.modalities[0]].index)
        for modality in self.modalities[1:]:
            sample_ids = sample_ids.intersection(set(self.omics_data[modality].index))
        
        if self.metadata is not None:
            sample_ids = sample_ids.intersection(set(self.metadata.index))
        
        sample_ids = sorted(list(sample_ids))
        
        if not sample_ids:
            raise ValueError("No common samples found across all modalities")
        
        print(f"Found {len(sample_ids)} common samples")
        
        # Align omics data
        for modality in self.modalities:
            data = self.omics_data[modality].loc[sample_ids].values.astype(np.float32)
            
            # Normalize if requested
            if self.normalize:
                # Standardize features (z-score normalization)
                data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            
            self.omics_data[modality] = data
        
        # Align metadata
        if self.metadata is not None:
            if self.cat_data is not None:
                self.cat_data = self.cat_data[self.metadata.index.isin(sample_ids)]
            if self.num_data is not None:
                self.num_data = self.num_data[self.metadata.index.isin(sample_ids)]
            if self.labels is not None:
                self.labels = self.labels[self.metadata.index.isin(sample_ids)]
        
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        output = {}
        
        # Add omics data
        for modality in self.modalities:
            output[modality] = torch.tensor(self.omics_data[modality][idx], dtype=torch.float32)
        
        # Add metadata
        if self.cat_data is not None or self.num_data is not None:
            meta_input = {}
            if self.cat_data is not None:
                meta_input["x_cat"] = torch.tensor(self.cat_data[idx], dtype=torch.long)
            if self.num_data is not None:
                meta_input["x_num"] = torch.tensor(self.num_data[idx], dtype=torch.float32)
            output["metadata"] = meta_input
        
        # Add labels
        if self.labels is not None:
            output["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return output
    
    def get_feature_names(self, modality: str) -> List[str]:
        """Get feature names for a specific modality."""
        return self.feature_names.get(modality, [])
    
    def get_sample_id(self, idx: int) -> str:
        """Get sample ID for a given index."""
        return self.sample_ids[idx]
    
    def get_categorical_encoders(self) -> Dict[str, np.ndarray]:
        """Get categorical variable encoders."""
        return self.cat_encoders
    
    def get_label_encoder(self) -> np.ndarray:
        """Get label encoder."""
        return getattr(self, 'label_encoder', None)
    
    def get_input_dims(self) -> Dict[str, int]:
        """Get input dimensions for each modality."""
        return {modality: data.shape[1] for modality, data in self.omics_data.items()}
    
    def get_metadata_dims(self) -> tuple:
        """Get metadata dimensions (cat_dims, num_dims)."""
        cat_dims = []
        if self.cat_data is not None:
            cat_dims = [len(encoder) for encoder in self.cat_encoders.values()]
        
        num_dims = 0
        if self.num_data is not None:
            num_dims = self.num_data.shape[1]
            
        return cat_dims, num_dims
