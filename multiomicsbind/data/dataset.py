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
                        if len(cat_data) == 1:
                            # Handle single categorical column - ensure 2D shape
                            self.cat_data = cat_data[0].reshape(-1, 1)
                        else:
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

    def _is_binary_data(self, data: np.ndarray) -> bool:
        """
        Detect if data is binary (only contains 0 and 1 values).
        
        Args:
            data: Numpy array to check
            
        Returns:
            True if data is binary, False otherwise
        """
        # Get unique values, excluding NaN
        unique_values = np.unique(data[~np.isnan(data)])
        
        # Check if all unique values are in {0, 1}
        # Handle both integer and float representations
        return len(unique_values) > 0 and set(unique_values).issubset({0, 0.0, 1, 1.0})
    
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
                # Check if data is binary (e.g., mutation data with 0/1 values)
                if self._is_binary_data(data):
                    print(f"  {modality}: Binary data detected - skipping normalization")
                else:
                    # Standardize features (z-score normalization)
                    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                    print(f"  {modality}: Normalized ({data.shape[1]} features)")
            
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


class TemporalMultiOmicsDataset(Dataset):
    """
    PyTorch Dataset for temporal multi-omics data.
    
    This dataset handles mixed scenarios where some modalities are static (single timepoint)
    and others are temporal (multiple timepoints). Common use case: transcriptomics and 
    cell painting at one timepoint, but proteomics measured across multiple timepoints.
    
    Args:
        static_data_paths (Dict[str, str]): Paths to static modality data files
        temporal_data_paths (Dict[str, str]): Paths to temporal modality data files
        temporal_metadata (Dict[str, Dict]): Metadata for temporal modalities including:
            - 'timepoints': List of timepoint identifiers
            - 'time_col': Column name for timepoint in data files
        metadata_path (Optional[str]): Path to sample metadata CSV file
        cat_cols (Optional[List[str]]): Categorical metadata columns
        num_cols (Optional[List[str]]): Numerical metadata columns
        label_col (Optional[str]): Label column name
        sample_id_col (str): Sample ID column name
        normalize (bool): Whether to normalize data
        max_seq_len (Optional[int]): Maximum sequence length for padding
        
    Example:
        >>> static_paths = {
        ...     'transcriptomics': 'tx_data.csv',
        ...     'cell_painting': 'cp_data.csv'
        ... }
        >>> temporal_paths = {
        ...     'proteomics': 'proteomics_timeseries.csv'
        ... }
        >>> temporal_meta = {
        ...     'proteomics': {
        ...         'timepoints': [0, 1, 2, 4, 8],  # hours
        ...         'time_col': 'timepoint'
        ...     }
        ... }
        >>> dataset = TemporalMultiOmicsDataset(
        ...     static_data_paths=static_paths,
        ...     temporal_data_paths=temporal_paths,
        ...     temporal_metadata=temporal_meta
        ... )
    """
    
    def __init__(
        self,
        static_data_paths: Dict[str, str],
        temporal_data_paths: Dict[str, str],
        temporal_metadata: Dict[str, Dict],
        metadata_path: Optional[str] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        sample_id_col: str = "sample_id",
        normalize: bool = True,
        max_seq_len: Optional[int] = None
    ):
        self.static_data_paths = static_data_paths
        self.temporal_data_paths = temporal_data_paths
        self.temporal_metadata = temporal_metadata
        self.metadata_path = metadata_path
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.label_col = label_col
        self.sample_id_col = sample_id_col
        self.normalize = normalize
        self.max_seq_len = max_seq_len
        
        # Storage for processed data
        self.static_data = {}
        self.temporal_data = {}
        self.static_feature_names = {}
        self.temporal_feature_names = {}
        self.sample_ids = []
        self.sequence_lengths = {}  # Track actual sequence lengths for each sample
        
        # Metadata storage
        self.cat_data = None
        self.num_data = None
        self.cat_encoders = {}
        self.labels = None
        self.label_encoder = None
        
        # Load and process data
        self._load_data()
    
    def _load_data(self):
        """Load and process both static and temporal data."""
        print("Loading temporal multi-omics dataset...")
        
        # Load static modalities
        static_dfs = {}
        for modality, path in self.static_data_paths.items():
            print(f"Loading static {modality} from {path}")
            df = pd.read_csv(path)
            static_dfs[modality] = df
            print(f"Loaded {modality}: {df.shape[0]} samples, {df.shape[1]-1} features")
        
        # Load temporal modalities
        temporal_dfs = {}
        for modality, path in self.temporal_data_paths.items():
            print(f"Loading temporal {modality} from {path}")
            df = pd.read_csv(path)
            temporal_dfs[modality] = df
            print(f"Loaded temporal {modality}: {df.shape[0]} timepoint measurements")
        
        # Find common samples across all modalities
        all_sample_sets = []
        
        # Add static modality samples
        for modality, df in static_dfs.items():
            all_sample_sets.append(set(df[self.sample_id_col].unique()))
        
        # Add temporal modality samples
        for modality, df in temporal_dfs.items():
            all_sample_sets.append(set(df[self.sample_id_col].unique()))
        
        # Get intersection of all sample sets
        common_samples = set.intersection(*all_sample_sets) if all_sample_sets else set()
        
        if not common_samples:
            raise ValueError("No common samples found across all modalities")
        
        self.sample_ids = sorted(list(common_samples))
        print(f"Found {len(self.sample_ids)} common samples")
        
        # Process static modalities
        for modality, df in static_dfs.items():
            # Filter to common samples
            df_filtered = df[df[self.sample_id_col].isin(common_samples)]
            
            # Sort by sample ID to ensure consistent ordering
            df_filtered = df_filtered.sort_values(self.sample_id_col)
            
            # Extract features (exclude sample ID column)
            feature_cols = [col for col in df_filtered.columns if col != self.sample_id_col]
            features = df_filtered[feature_cols].values.astype(np.float32)
            
            # Normalize if requested
            if self.normalize:
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            self.static_data[modality] = features
            self.static_feature_names[modality] = feature_cols
        
        # Process temporal modalities
        for modality, df in temporal_dfs.items():
            # Get temporal metadata
            temporal_meta = self.temporal_metadata[modality]
            timepoints = temporal_meta['timepoints']
            time_col = temporal_meta['time_col']
            
            # Filter to common samples
            df_filtered = df[df[self.sample_id_col].isin(common_samples)]
            
            # Get feature columns (exclude sample ID and time columns)
            feature_cols = [col for col in df_filtered.columns 
                          if col not in [self.sample_id_col, time_col]]
            
            # Initialize temporal data array
            n_samples = len(self.sample_ids)
            n_timepoints = len(timepoints)
            n_features = len(feature_cols)
            
            temporal_array = np.full((n_samples, n_timepoints, n_features), np.nan, dtype=np.float32)
            sequence_lengths = np.zeros(n_samples, dtype=int)
            
            # Fill temporal data
            for sample_idx, sample_id in enumerate(self.sample_ids):
                sample_data = df_filtered[df_filtered[self.sample_id_col] == sample_id]
                
                valid_timepoints = 0
                for time_idx, timepoint in enumerate(timepoints):
                    time_data = sample_data[sample_data[time_col] == timepoint]
                    
                    if not time_data.empty:
                        # Take first row if multiple (shouldn't happen in well-formed data)
                        features = time_data[feature_cols].iloc[0].values.astype(np.float32)
                        temporal_array[sample_idx, time_idx, :] = features
                        valid_timepoints += 1
                
                sequence_lengths[sample_idx] = valid_timepoints
            
            # Normalize if requested (normalize across all timepoints and samples)
            if self.normalize:
                # Reshape for normalization (flatten time and sample dimensions)
                orig_shape = temporal_array.shape
                flat_data = temporal_array.reshape(-1, n_features)
                
                # Only normalize non-NaN values
                valid_mask = ~np.isnan(flat_data)
                for feat_idx in range(n_features):
                    feat_valid = valid_mask[:, feat_idx]
                    if feat_valid.sum() > 0:
                        feat_data = flat_data[feat_valid, feat_idx]
                        mean_val = feat_data.mean()
                        std_val = feat_data.std()
                        
                        # Normalize only valid values
                        flat_data[feat_valid, feat_idx] = (feat_data - mean_val) / (std_val + 1e-8)
                
                temporal_array = flat_data.reshape(orig_shape)
            
            self.temporal_data[modality] = temporal_array
            self.temporal_feature_names[modality] = feature_cols
            self.sequence_lengths[modality] = sequence_lengths
        
        # Load metadata if provided
        if self.metadata_path:
            self._load_metadata()
        
        print(f"Dataset initialized with {len(self.sample_ids)} samples")
        print(f"Static modalities: {list(self.static_data.keys())}")
        print(f"Temporal modalities: {list(self.temporal_data.keys())}")
    
    def _load_metadata(self):
        """Load and process metadata."""
        meta_df = pd.read_csv(self.metadata_path)
        
        # Filter to common samples
        meta_df = meta_df[meta_df[self.sample_id_col].isin(self.sample_ids)]
        meta_df = meta_df.sort_values(self.sample_id_col)
        
        # Process categorical variables
        if self.cat_cols:
            cat_data = []
            for col in self.cat_cols:
                if col in meta_df.columns:
                    unique_vals = meta_df[col].unique()
                    encoder = {val: idx for idx, val in enumerate(unique_vals)}
                    self.cat_encoders[col] = unique_vals
                    
                    encoded = meta_df[col].map(encoder).values
                    cat_data.append(encoded)
                else:
                    warnings.warn(f"Categorical column '{col}' not found in metadata")
            
            if cat_data:
                if len(cat_data) == 1:
                    # Handle single categorical column - ensure 2D shape
                    self.cat_data = cat_data[0].reshape(-1, 1).astype(np.int64)
                else:
                    self.cat_data = np.column_stack(cat_data).astype(np.int64)
        
        # Process numerical variables
        if self.num_cols:
            num_data = []
            for col in self.num_cols:
                if col in meta_df.columns:
                    num_data.append(meta_df[col].values)
                else:
                    warnings.warn(f"Numerical column '{col}' not found in metadata")
            
            if num_data:
                self.num_data = np.column_stack(num_data).astype(np.float32)
        
        # Process labels
        if self.label_col and self.label_col in meta_df.columns:
            unique_labels = meta_df[self.label_col].unique()
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
            self.labels = meta_df[self.label_col].map(self.label_encoder).values.astype(np.int64)
            print(f"Labels: {len(unique_labels)} classes - {unique_labels}")
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample containing both static and temporal data."""
        output = {}
        
        # Add static modalities
        for modality, data in self.static_data.items():
            output[modality] = torch.tensor(data[idx], dtype=torch.float32)
        
        # Add temporal modalities
        for modality, data in self.temporal_data.items():
            temporal_sample = data[idx]  # Shape: (seq_len, features)
            seq_len = self.sequence_lengths[modality][idx]
            
            # Create mask for valid timepoints
            mask = torch.zeros(temporal_sample.shape[0], dtype=torch.bool)
            mask[:seq_len] = True
            
            output[f"{modality}"] = torch.tensor(temporal_sample, dtype=torch.float32)
            output[f"{modality}_mask"] = mask
            output[f"{modality}_seq_len"] = seq_len
        
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
    
    def get_input_dims(self) -> Dict[str, int]:
        """Get input dimensions for each modality."""
        dims = {}
        
        # Static modalities
        for modality, data in self.static_data.items():
            dims[modality] = data.shape[1]
        
        # Temporal modalities
        for modality, data in self.temporal_data.items():
            dims[modality] = data.shape[2]  # Last dimension is features
        
        return dims
    
    def get_temporal_info(self) -> Dict[str, Dict]:
        """Get temporal information for each temporal modality."""
        temporal_info = {}
        
        for modality in self.temporal_data.keys():
            temporal_info[modality] = {
                'max_seq_len': self.temporal_data[modality].shape[1],
                'n_features': self.temporal_data[modality].shape[2],
                'timepoints': self.temporal_metadata[modality]['timepoints']
            }
        
        return temporal_info
    
    def get_metadata_dims(self) -> tuple:
        """Get metadata dimensions (cat_dims, num_dims)."""
        cat_dims = []
        if self.cat_data is not None:
            cat_dims = [len(encoder) for encoder in self.cat_encoders.values()]
        
        num_dims = 0
        if self.num_data is not None:
            num_dims = self.num_data.shape[1]
            
        return cat_dims, num_dims
    
    def get_feature_names(self, modality: str) -> List[str]:
        """Get feature names for a specific modality."""
        if modality in self.static_feature_names:
            return self.static_feature_names[modality]
        elif modality in self.temporal_feature_names:
            return self.temporal_feature_names[modality]
        else:
            return []
