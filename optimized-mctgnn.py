import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import kagglehub
import os
import logging
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings
import gc
import psutil

# Logging and device setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    raise RuntimeError("CUDA is not available. Please set runtime to GPU.")

# Enhanced constants for H100 GPU with 80GB VRAM
NUM_NODES = 162
IN_CHANNELS = 1
MAX_SEQ_LENGTH = 5000
BATCH_SIZE = 8  # Increased from 2 to better utilize H100
LEARNING_RATE = 0.0005
EPOCHS = 50
HIDDEN_DIM = 64  # Increased from 32 to utilize more capacity
SUBSAMPLE_FACTOR = 5  # Reduced from 10 for more accurate integration

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True

# Helper Functions
def parse_numeric_string(s):
    try:
        if not isinstance(s, str):
            return None
        s = s.strip('[]').replace('\n', ' ').replace('...', '0').strip()
        if not s or s.isspace():
            return None
        values = [float(x) for x in s.split() if x.strip() and x.replace('.', '').replace('-', '').isdigit()]
        return np.array(values) if values else None
    except Exception as e:
        logger.warning(f"Failed to parse string: {s[:30]}... - Error: {str(e)}")
        return None

# Dataset Class with optimized processing for large RAM
class NeuralSignalDataset(Dataset):
    def __init__(self, data_df, coords_df, max_seq_length=MAX_SEQ_LENGTH, normalize=True,
                 chunk_min_size=20, max_chunks=200):  # Increased max_chunks to use more data
        self.data = data_df
        self.coords = coords_df
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.chunk_min_size = chunk_min_size
        self.max_chunks = max_chunks
        self.valid_indices = [i for i in range(194) if i not in [0, 16, 23, 24, 26, 31, 38, 63, 92, 96, 99, 100, 101, 102, 108, 111, 112, 113, 114, 122, 128, 129, 139, 142, 145, 146, 147, 148, 170, 177, 192, 193]]
        self.process_data()

    def process_data(self):
        """Process the neural data into chunks based on stimulus changes with parallel processing."""
        print("Chunking data dynamically based on stimulus changes...")
        total_rows = len(self.data)
        chunked_data = []
        actual_lengths = []
        current_activity = None
        current_chunk = []

        # Use larger buffer for faster processing with more RAM
        buffer_size = 10000  # Process data in larger chunks
        
        # Process row by row with optimized chunking
        processed_count = 0
        skipped_count = 0
        chunk_count = 0

        # Process in buffer chunks to utilize RAM efficiently
        for buffer_start in range(0, total_rows, buffer_size):
            buffer_end = min(buffer_start + buffer_size, total_rows)
            buffer_data = self.data.iloc[buffer_start:buffer_end]
            
            progress_bar = tqdm(range(len(buffer_data)), 
                               desc=f"Processing rows {buffer_start}-{buffer_end-1}")
            
            for idx in progress_bar:
                try:
                    electrode_str = buffer_data.iloc[idx]['data']
                    electrode_values = parse_numeric_string(electrode_str)
                    if electrode_values is None or len(electrode_values) != 194:
                        skipped_count += 1
                        continue
                    
                    electrode_values = electrode_values[self.valid_indices]
                    if len(electrode_values) != NUM_NODES:
                        skipped_count += 1
                        continue

                    activity = buffer_data.iloc[idx]['activity']
                    processed_count += 1

                    if current_activity is None:
                        current_activity = activity
                        current_chunk = [electrode_values]
                    elif activity != current_activity:
                        if len(current_chunk) > self.chunk_min_size and len(chunked_data) < self.max_chunks:
                            chunked_data.append(np.vstack(current_chunk))
                            actual_lengths.append(len(current_chunk))
                            chunk_count += 1
                        current_chunk = [electrode_values]
                        current_activity = activity
                    else:
                        current_chunk.append(electrode_values)

                    # Update progress bar description with stats
                    if idx % 100 == 0:
                        progress_bar.set_postfix({
                            'processed': processed_count,
                            'skipped': skipped_count,
                            'chunks': chunk_count,
                            'current_chunk_size': len(current_chunk)
                        })

                except Exception as e:
                    logger.error(f"Error processing row {buffer_start + idx}: {str(e)}")
                    skipped_count += 1
                    continue

        # Process the last chunk if needed
        if current_chunk and len(current_chunk) > self.chunk_min_size and len(chunked_data) < self.max_chunks:
            chunked_data.append(np.vstack(current_chunk))
            actual_lengths.append(len(current_chunk))
            chunk_count += 1

        if not chunked_data:
            raise ValueError("No valid data chunks processed.")

        print(f"Processing complete! Found {processed_count} valid rows, created {chunk_count} chunks")

        # Setup progress bar for padding and processing
        print("Preparing final dataset...")
        self.electrode_data = []
        self.actual_lengths = []

        # Process in parallel with larger batch size
        for chunk_idx, (chunk, actual_length) in enumerate(zip(chunked_data, actual_lengths)):
            print(f"Processing chunk {chunk_idx+1}/{len(chunked_data)}, length={actual_length}")
            if actual_length > self.max_seq_length:
                self.electrode_data.append(chunk[:self.max_seq_length])
                self.actual_lengths.append(self.max_seq_length)
            else:
                pad_len = self.max_seq_length - actual_length
                self.electrode_data.append(np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant'))
                self.actual_lengths.append(actual_length)

        # Using NumPy's optimized stacking for large arrays
        self.electrode_data = np.stack(self.electrode_data)
        print(f"Stacked data shape before normalization: {self.electrode_data.shape}")

        if self.normalize:
            print("Normalizing data...")
            try:
                # Process in chunks to handle large arrays efficiently
                batch_size = 10  # Process 10 chunks at a time
                for i in range(0, len(self.electrode_data), batch_size):
                    batch_end = min(i + batch_size, len(self.electrode_data))
                    batch = self.electrode_data[i:batch_end]
                    
                    reshaped_batch = batch.reshape(-1, NUM_NODES)
                    scaler = StandardScaler()
                    normalized_batch = scaler.fit_transform(reshaped_batch)
                    self.electrode_data[i:batch_end] = normalized_batch.reshape(batch.shape)
                    
                    # Force garbage collection after each batch
                    del batch, reshaped_batch, normalized_batch, scaler
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error during normalization: {str(e)}")
                # If standard scaling fails, use a simpler approach
                mean = np.mean(self.electrode_data)
                std = np.std(self.electrode_data) + 1e-8
                self.electrode_data = (self.electrode_data - mean) / std

        print(f"Dataset preparation complete! Final shape: {self.electrode_data.shape}")

    def __len__(self):
        return len(self.electrode_data)

    def __getitem__(self, idx):
        sequences = self.electrode_data[idx]
        actual_length = self.actual_lengths[idx]
        return torch.FloatTensor(sequences).unsqueeze(-1), torch.tensor(actual_length, dtype=torch.long)

# Optimized GNN Function for ODE integration
class StableGNNFunc(nn.Module):
    def __init__(self, hidden_dim, adj_matrix):
        super(StableGNNFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("adj_matrix", adj_matrix.clone())  # Register as buffer for better CUDA management

        # Multi-layer GNN with additional capacity
        self.gnn_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),  # Wider network
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),  # SiLU is smoother than ReLU
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Activity-state dynamics with deeper networks
        self.dx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded activation for stability
        )

        # Memory-state dynamics with deeper networks
        self.dm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded activation for stability
        )

        # Coefficients tuned for neuronal dynamics
        self.diffusion_coef = 0.02  # Spatial spread parameter
        self.decay_coef = 0.03     # Neural decay parameter
        self.scale_factor = 2.0    # Overall dynamics speed

    def forward(self, t, z):
        # Get shape info
        batch_size, num_nodes, total_dim = z.shape
        device_z = z.device

        # Make sure tensors are on the correct device
        if self.adj_matrix.device != device_z:
            self.adj_matrix = self.adj_matrix.to(device_z)

        assert total_dim == 2 * self.hidden_dim, f"Expected dim {2 * self.hidden_dim}, got {total_dim}"

        # State decomposition
        x = z[:, :, :self.hidden_dim]  # Neural activity state
        m = z[:, :, self.hidden_dim:]  # Memory component

        # Check for NaN inputs
        if torch.isnan(z).any():
            x = torch.nan_to_num(x, nan=0.0)
            m = torch.nan_to_num(m, nan=0.0)

        # Concatenate states
        z_cat = torch.cat([x, m], dim=-1)

        # Apply GNN layer
        transformed = self.gnn_layer(z_cat)

        # Normalize to prevent extreme values
        transformed = F.normalize(transformed, p=2, dim=-1) * (self.hidden_dim**0.5)

        # Graph diffusion - node interactions through adjacency matrix
        neighbor_influence = torch.matmul(self.adj_matrix.float(), transformed)

        # Calculate dynamics with scale factor
        dx_dt = self.scale_factor * self.dx_head(
            transformed + self.diffusion_coef * neighbor_influence - self.decay_coef * x
        )
        dm_dt = self.scale_factor * self.dm_head(
            transformed + self.diffusion_coef * neighbor_influence
        )

        # Combine state derivatives
        dz_dt = torch.cat([dx_dt, dm_dt], dim=-1)

        # Gradient norm clipping for stability
        norm = torch.norm(dz_dt, dim=-1, keepdim=True)
        max_norm = 10.0  # Max gradient magnitude
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        dz_dt = dz_dt * scale

        # Safety check for NaN and replace if necessary
        if torch.isnan(dz_dt).any():
            dz_dt = torch.nan_to_num(dz_dt, nan=0.0)

        return dz_dt

# Neural ODE with cubic spline interpolation
class StableNeuralODEGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, adj_matrix):
        super(StableNeuralODEGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("adj_matrix", adj_matrix.clone())  # Register as buffer

        # Deeper encoder network for better initial conditions
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 2 * hidden_dim)
        )

        self.ode_func = StableGNNFunc(hidden_dim, adj_matrix)

    def cubic_spline_interpolation(self, solution, integration_time, full_time, batch_size, num_nodes):
        """Apply cubic spline interpolation to the ODE solution."""
        print("Performing cubic spline interpolation...")
        
        # Move to CPU for scipy interpolation (necessary as scipy doesn't work on GPU)
        solution_cpu = solution.detach().cpu().numpy()
        integration_time_cpu = integration_time.detach().cpu().numpy()
        full_time_cpu = full_time.detach().cpu().numpy()
        
        # Create the output buffer
        full_solution = torch.zeros(
            len(full_time_cpu), batch_size, num_nodes, 2 * self.hidden_dim, 
            device=solution.device
        )
        
        # Process in batches to efficiently use memory
        batch_increment = 2  # Process 2 samples at a time
        for b_start in range(0, batch_size, batch_increment):
            b_end = min(b_start + batch_increment, batch_size)
            
            # Process each node for this batch
            for n in range(num_nodes):
                # Process each feature dimension
                for d in range(2 * self.hidden_dim):
                    # Create spline for each feature in current batch/node
                    for b in range(b_start, b_end):
                        # Extract values for this specific batch, node, dimension
                        y_values = solution_cpu[:, b, n, d]
                        
                        # Create cubic spline interpolator
                        cs = CubicSpline(integration_time_cpu, y_values)
                        
                        # Interpolate at desired timepoints
                        interpolated = cs(full_time_cpu)
                        
                        # Store results back in tensor
                        full_solution[:, b, n, d] = torch.tensor(
                            interpolated, device=solution.device
                        )
        
        print("Cubic spline interpolation complete.")
        return full_solution

    def forward(self, x):
        batch_size, seq_len, num_nodes, channels = x.shape

        # Get initial state from first time step
        initial_x = x[:, 0, :, :]
        z0 = self.encoder(initial_x)

        # Create integration time points
        integration_time = torch.linspace(0, seq_len-1, seq_len//SUBSAMPLE_FACTOR + 1, dtype=torch.float32).to(device)

        print(f"Starting ODE integration with {len(integration_time)} time points...")

        # Suppress warnings during integration
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            try:
                # Use dopri5 solver with careful tolerance settings
                solution = odeint(
                    self.ode_func,
                    z0,
                    integration_time,
                    method='dopri5',
                    rtol=1e-3,
                    atol=1e-4,
                    options={'max_num_steps': 10000}  # Allow more steps for complex dynamics
                )

                print(f"ODE integration complete. Solution shape: {solution.shape}")

                # Check for NaN values in the solution
                if torch.isnan(solution).any():
                    print("Warning: NaN values detected in ODE solution. Replacing with zeros.")
                    solution = torch.nan_to_num(solution, nan=0.0)

                # Create full time points for interpolation
                full_time = torch.arange(seq_len, dtype=torch.float32).to(device)
                
                # Apply cubic spline interpolation (our new method)
                full_solution = self.cubic_spline_interpolation(
                    solution, integration_time, full_time, batch_size, num_nodes
                )

                return full_solution

            except Exception as e:
                logger.error(f"ODE integration failed: {str(e)}")
                print(f"Error during dopri5 integration: {str(e)}. Falling back to euler method.")

                # Fall back to euler method
                try:
                    solution = odeint(
                        self.ode_func,
                        z0,
                        integration_time,
                        method='euler',
                        rtol=1e-2,
                        atol=1e-3
                    )

                    # Create full time points for interpolation
                    full_time = torch.arange(seq_len, dtype=torch.float32).to(device)
                    
                    # Apply cubic spline interpolation
                    full_solution = self.cubic_spline_interpolation(
                        solution, integration_time, full_time, batch_size, num_nodes
                    )
                    
                    return full_solution

                except Exception as e2:
                    logger.error(f"Fallback integration also failed: {str(e2)}")
                    print("Using simple time-stepping as final fallback")

                    # Ultimate fallback - simple forward evolution
                    full_solution = torch.zeros(seq_len, batch_size, num_nodes, 2 * self.hidden_dim, device=device)
                    state = z0
                    full_solution[0] = state

                    for t in range(1, seq_len):
                        # Simple Euler step
                        derivative = self.ode_func(torch.tensor(float(t), device=device), state)
                        state = state + derivative * 1.0  # dt = 1.0
                        full_solution[t] = state

                        # Periodically clean up memory
                        if t % 500 == 0:
                            torch.cuda.empty_cache()

                    return full_solution

# Enhanced Temporal GNN - with increased capacity
class EnhancedTemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, adj_matrix):
        super(EnhancedTemporalGNN, self).__init__()
        self.neural_ode_gnn = StableNeuralODEGNN(in_channels, hidden_dim, adj_matrix)

        # Enhanced multi-layer decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )

    def forward(self, x):
        # Get ODE solution
        solution = self.neural_ode_gnn(x)

        # Extract activity state
        pred_x = solution[:, :, :, :self.neural_ode_gnn.hidden_dim]

        # Apply decoder to get predictions
        pred_observations = self.decoder(pred_x)

        # Permute to match expected shape
        pred_sequences = pred_observations.permute(1, 0, 2, 3)

        return pred_sequences, solution

# Create optimized correlation adjacency matrix
def create_correlation_adjacency_matrix(data, valid_indices, n_samples=100000, threshold=0.5):
    """Creates optimized correlation adjacency matrix using more data points"""
    print(f"Computing correlation adjacency matrix across {n_samples} timestamps...")

    # Process in chunks to optimize memory usage
    chunk_size = 5000
    sample_data_chunks = []
    
    for chunk_start in range(0, min(n_samples, len(data)), chunk_size):
        chunk_end = min(chunk_start + chunk_size, min(n_samples, len(data)))
        print(f"Processing correlation chunk {chunk_start}-{chunk_end}")
        
        chunk_samples = []
        for idx in tqdm(range(chunk_start, chunk_end), desc=f"Building correlation matrix chunk {chunk_start}-{chunk_end}"):
            try:
                electrode_str = data.iloc[idx]['data']
                electrode_values = parse_numeric_string(electrode_str)
                if electrode_values is not None and len(electrode_values) == 194:
                    chunk_samples.append(electrode_values[valid_indices])
            except Exception as e:
                logger.error(f"Error processing index {idx}: {str(e)}")
                continue
                
        if chunk_samples:
            sample_data_chunks.append(np.array(chunk_samples).T)
            
    # Combine chunks efficiently
    if not sample_data_chunks:
        raise ValueError("No valid samples collected for adjacency matrix")
    
    # Compute correlations in chunks for memory efficiency
    corr_matrices = []
    
    for chunk in sample_data_chunks:
        chunk_corr = np.corrcoef(chunk)
        chunk_corr = np.nan_to_num(chunk_corr)
        corr_matrices.append(chunk_corr)
    
    # Average the correlation matrices
    corr_matrix = np.mean(corr_matrices, axis=0)
    
    print(f"Creating adjacency matrix with threshold {threshold}...")
    adj_matrix = (np.abs(corr_matrix) > threshold).astype(np.float32)
    np.fill_diagonal(adj_matrix, 0)

    edges = adj_matrix.sum()
    print(f"Created adjacency matrix with {edges} edges")

    return torch.FloatTensor(adj_matrix).to(device)

# Main function with optimized training for H100
def main():
    try:
        os.makedirs("results", exist_ok=True)
        
        # Add memory reporting for better monitoring
        print("\n===== SYSTEM RESOURCES =====")
        print(f"CPU RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print("===========================\n")

        # Optimize performance for H100
        torch.cuda.empty_cache()  # Clear GPU cache before starting
        
        # Set larger limits for GPU operations
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM

        # Data loading
        print("Loading electrode coordinates...")
        coords_path = kagglehub.dataset_download("arunramponnambalam/electrodes-coordinates")
        coords_file = os.path.join(coords_path, "electrodesdata.csv")
        coords = pd.read_csv(coords_file, usecols=[0, 1, 2], names=['x', 'y', 'z'], header=0)
        valid_indices = [i for i in range(len(coords)) if i not in [0, 16, 23, 24, 26, 31, 38, 63, 92, 96, 99, 100, 101, 102, 108, 111, 112, 113, 114, 122, 128, 129, 139, 142, 145, 146, 147, 148, 170, 177, 192, 193]]
        coords = coords.iloc[valid_indices].reset_index(drop=True)

        print("Loading neural activity data...")
        data_path = kagglehub.dataset_download("nocopyrights/image-stimulus-timestamp-data")
        data_file = os.path.join(data_path, "merged_data.csv")
        data = pd.read_csv(data_file)
        print(f"Loaded dataset with {len(data)} samples")

        data = data[~(data['activity'].str.startswith('sscr', na=False) | data['activity'].str.startswith('sscr2', na=False))].reset_index(drop=True)
        print(f"Dataset after filtering: {len(data)} samples")

        print("Creating adjacency matrix...")
        adj_matrix = create_correlation_adjacency_matrix(data, valid_indices)

        print("Initializing dataset...")
        dataset = NeuralSignalDataset(data, coords)

        # Free up memory
        del data
        gc.collect()
        torch.cuda.empty_cache()

        print("Creating data loader...")
        try:
            # Use more workers to better utilize CPU cores
            train_loader = DataLoader(
                dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=6,  # Use more CPU cores
                pin_memory=True,
                prefetch_factor=4  # Prefetch more batches
            )
        except Exception as e:
            print(f"Error with multiple workers: {str(e)}, trying with 2 workers")
            train_loader = DataLoader(
                dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=2, 
                pin_memory=True
            )

        # Use mixed precision training for speed
        scaler = torch.cuda.amp.GradScaler()

        # Model initialization and training setup
        print("Initializing model...")
        model = EnhancedTemporalGNN(IN_CHANNELS, HIDDEN_DIM, adj_matrix).to(device)
        
        # Print model size and parameters
        model_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {model_params:,} parameters")

        # Use AdamW with carefully tuned parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-6,
            eps=1e-8
        )

        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE * 10,
            total_steps=EPOCHS * len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Keep track of best model
        best_loss = float('inf')
        best_model_state = None

        # Training loop
        print("Starting training...")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            batch_count = 0

            # Training loop with progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for sequences, actual_lengths in train_pbar:
                try:
                    sequences = sequences.to(device)
                    actual_lengths = actual_lengths.to(device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Use automatic mixed precision for faster training
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        pred_sequences, _ = model(sequences)

                        # Create mask for sequence lengths
                        batch_size, seq_len, _, _ = sequences.shape
                        mask = torch.zeros(batch_size, seq_len, device=device)
                        for b in range(batch_size):
                            mask[b, :actual_lengths[b]] = 1

                        # Masked loss calculation
                        loss = ((pred_sequences - sequences) ** 2 * mask.unsqueeze(-1).unsqueeze(-1)).sum() / (mask.sum() * NUM_NODES * IN_CHANNELS)

                        # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Invalid loss detected, skipping batch")
                        # Try to recover by reinitializing the last batch
                        torch.cuda.empty_cache()
                        continue

                    # Backward pass and optimization with mixed precision
                    scaler.scale(loss).backward()

                    # Conservative gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # Update statistics
                    total_loss += loss.item()
                    batch_count += 1

                    # Update progress bar
                    train_pbar.set_postfix({
                        'loss': loss.item(), 
                        'lr': optimizer.param_groups[0]['lr']
                    })

                    # Free GPU memory periodically
                    if batch_count % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error during training batch: {str(e)}")
                    torch.cuda.empty_cache()  # Clean up after error
                    continue

            # Calculate average loss
            avg_loss = total_loss / max(1, batch_count)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Reconstruction Loss: {avg_loss:.4f}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_loss
                }
                print(f"New best model saved! Loss: {best_loss:.4f}")
                # Save checkpoint for recovery
                torch.save(best_model_state, f"results/ecog_ode_model_checkpoint_epoch{epoch}.pt")