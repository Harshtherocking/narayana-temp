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
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
import gc
from scipy.interpolate import CubicSpline
import time
from sklearn.decomposition import PCA
from torchdiffeq import odeint
import seaborn as sns
import glob
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

# Logging and device setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    # Optimizations for GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Display available GPU memory
    free_memory, total_memory = torch.cuda.mem_get_info()
    print(f"GPU Memory: {free_memory/(1024**3):.2f}GB free / {total_memory/(1024**3):.2f}GB total")
else:
    print("CUDA is not available. Using CPU but performance will be slow.")

# ===== OPTIMIZED CONSTANTS =====
NUM_NODES = 162  # Number of valid electrodes
IN_CHANNELS = 1
MAX_SEQ_LENGTH = 4100  # Reduced from 5000 to save memory
BATCH_SIZE = 2         # Reduced from 4 to save memory
LEARNING_RATE = 0.0003
EPOCHS = 10
HIDDEN_DIM = 32        # Reduced from 64 to save memory
SUBSAMPLE_FACTOR = 8   # Reduced from 10 for better resolution
MEMORY_ENABLED = True # Disabled memory state temporarily to reduce complexity
USE_MIXED_PRECISION = True  # Use mixed precision for memory efficiency
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients to simulate larger batch
MAX_CHUNKS = 65        # Limit dataset size for faster training

# Create results directory
os.makedirs("results", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

# Helper Functions
def parse_numeric_string(s):
    """Parse string representations of numeric arrays from the dataset."""
    try:
        if not isinstance(s, str):
            return None
        s = s.strip('[]').replace('\n', ' ').replace('...', '0').strip()
        if not s or s.isspace():
            return None
        values = [float(x) for x in s.split() if x.strip() and x.replace('.', '').replace('-', '').isdigit()]
        return np.array(values) if values else None
    except Exception as e:
        return None

# Dataset Class with improved progress tracking and memory efficiency
class NeuralSignalDataset(Dataset):
    """Enhanced dataset for processing neural signal data with efficient chunking."""
    def __init__(self, data_df, coords_df, max_seq_length=MAX_SEQ_LENGTH, normalize=True,
                 chunk_min_size=20, max_chunks=MAX_CHUNKS, subsample_factor=1):
        self.data = data_df
        self.coords = coords_df
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.chunk_min_size = chunk_min_size
        self.max_chunks = max_chunks
        self.subsample_factor = subsample_factor

        # List of valid electrode indices (removing problematic electrodes)
        self.valid_indices = [i for i in range(194) if i not in
                            [0, 16, 23, 24, 26, 31, 38, 63, 92, 96, 99, 100, 101, 102,
                             108, 111, 112, 113, 114, 122, 128, 129, 139, 142, 145, 146,
                             147, 148, 170, 177, 192, 193]]

        self.process_data()

    def process_data(self):
        """Process the neural data into chunks based on stimulus changes."""
        print("Chunking data dynamically based on stimulus changes...")
        total_rows = len(self.data)
        chunked_data = []
        actual_lengths = []
        current_activity = None
        current_chunk = []

        # Setup progress bar
        progress_bar = tqdm(range(total_rows), desc="Processing rows")

        # Process row by row with original chunking logic
        processed_count = 0
        skipped_count = 0
        chunk_count = 0

        for idx in progress_bar:
            try:
                # Apply subsampling - only process every nth row
                if idx % self.subsample_factor != 0:
                    continue

                electrode_str = self.data.iloc[idx]['data']
                electrode_values = parse_numeric_string(electrode_str)
                if electrode_values is None or len(electrode_values) != 194:
                    skipped_count += 1
                    continue
                electrode_values = electrode_values[self.valid_indices]
                if len(electrode_values) != NUM_NODES:
                    skipped_count += 1
                    continue

                activity = self.data.iloc[idx]['activity']
                processed_count += 1

                if current_activity is None:
                    current_activity = activity
                    current_chunk = [electrode_values]
                elif activity != current_activity or len(current_chunk) >= self.max_seq_length:
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

                # Break early if we've reached our chunk limit
                if chunk_count >= self.max_chunks:
                    break

            except Exception as e:
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

        # Clear intermediate data
        del current_chunk
        gc.collect()

        # Setup progress bar for padding and processing
        print("Preparing final dataset...")
        self.electrode_data = []
        self.actual_lengths = []

        for chunk_idx, (chunk, actual_length) in enumerate(zip(chunked_data, actual_lengths)):
            print(f"Processing chunk {chunk_idx+1}/{len(chunked_data)}, length={actual_length}")
            # Trim to max sequence length first if needed
            if actual_length > self.max_seq_length:
                chunk = chunk[:self.max_seq_length]
                actual_length = self.max_seq_length

            # Apply padding if needed
            pad_len = self.max_seq_length - actual_length
            padded_chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant')

            self.electrode_data.append(padded_chunk)
            self.actual_lengths.append(actual_length)

        self.electrode_data = np.stack(self.electrode_data)
        print(f"Stacked data shape before normalization: {self.electrode_data.shape}")

        if self.normalize:
            print("Normalizing data...")
            try:
                # Process in batches to save memory - use simpler normalization
                mean = np.mean(self.electrode_data)
                std = np.std(self.electrode_data) + 1e-8
                self.electrode_data = (self.electrode_data - mean) / std
            except Exception as e:
                logger.error(f"Error during normalization: {str(e)}")
                # If normalization fails, use simpler approach
                self.electrode_data = self.electrode_data / (np.max(np.abs(self.electrode_data)) + 1e-8)

        # Convert to float32 to save memory
        self.electrode_data = self.electrode_data.astype(np.float32)
        print(f"Dataset preparation complete! Final shape: {self.electrode_data.shape}")

    def __len__(self):
        return len(self.electrode_data)

    def __getitem__(self, idx):
        sequences = self.electrode_data[idx]
        actual_length = self.actual_lengths[idx]
        return torch.FloatTensor(sequences).unsqueeze(-1), torch.tensor(actual_length, dtype=torch.long)


# Optimized ODE Function
class OptimizedODEFunc(nn.Module):
    """Memory-efficient ODE function for neural dynamics."""
    def __init__(self, hidden_dim, adj_matrix):
        super(OptimizedODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.adj_matrix = adj_matrix

        # Convert adjacency matrix to sparse format for memory efficiency
        if not torch.is_sparse(self.adj_matrix):
            self.adj_matrix = self.adj_matrix.to_sparse()

        # Compute normalized adjacency only once and cache it
        self._compute_normalized_adj()

        # Main neural network for dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded activation for stability
        )

        # Attention mechanism for spatial influence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # Attention across nodes
        )

        # Learnable parameters
        self.diffusion_scale = nn.Parameter(torch.Tensor([0.01]))
        self.time_scale = nn.Parameter(torch.Tensor([1.0]))

    def _compute_normalized_adj(self):
        """Precompute normalized adjacency matrix."""
        # Convert to dense for normalization calculation
        adj_dense = self.adj_matrix.to_dense()
        node_degrees = adj_dense.sum(dim=1).view(-1, 1) + 1e-8
        self.normalized_adj = (adj_dense / node_degrees).to_sparse()

    def spatial_diffusion(self, x, batch_size):
        """Memory-efficient spatial diffusion calculation."""
        # Process in chunks to save memory
        chunk_size = 4  # Process 4 samples at a time
        results = []

        for i in range(0, batch_size, chunk_size):
            end_i = min(i + chunk_size, batch_size)
            chunk = x[i:end_i]
            chunk_size_actual = chunk.size(0)

            # Compute attention for this chunk
            attention = self.attention(chunk)
            x_weighted = chunk * attention

            # Use sparse matrix multiplication for memory efficiency
            chunk_diffusion = torch.stack([
                torch.sparse.mm(self.normalized_adj, x_weighted[b])
                for b in range(chunk_size_actual)
            ])

            results.append(chunk_diffusion * self.diffusion_scale)

        return torch.cat(results, dim=0)

    def forward(self, t, x):
        """Forward pass with memory optimizations."""
        batch_size, num_nodes, state_dim = x.shape

        # Handle NaN values
        x = torch.nan_to_num(x)

        # Process neural dynamics
        dynamics = self.dynamics_net(x)

        # Add spatial diffusion component
        diffusion = self.spatial_diffusion(x, batch_size)

        # Combine for final derivatives
        dx_dt = dynamics + diffusion

        # Apply time scaling and ensure stability
        dx_dt = self.time_scale * dx_dt

        # Gradient clipping for numerical stability
        norm = torch.norm(dx_dt, dim=-1, keepdim=True)
        max_norm = 10.0
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        dx_dt = dx_dt * scale

        # Final check for NaN values
        dx_dt = torch.nan_to_num(dx_dt)

        # Clear intermediate tensors
        del dynamics, diffusion, norm, scale

        return dx_dt


# Memory-efficient Neural ODE model
class OptimizedNeuralODE(nn.Module):
    """Neural ODE model with memory optimizations and stable integration."""
    def __init__(self, in_channels, hidden_dim, adj_matrix):
        super(OptimizedNeuralODE, self).__init__()
        self.hidden_dim = hidden_dim
        self.adj_matrix = adj_matrix
        self.integration_points = None  # Will be set during training

        # Encoder for initial state
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # ODE function
        self.ode_func = OptimizedODEFunc(hidden_dim, adj_matrix)

        # Decoder for predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_channels)
        )

        # Default solver settings
        self.solver = "rk4"  # Use RK4 by default as it's more stable than dopri5
        self.solver_rtol = 1e-2
        self.solver_atol = 1e-3
        self.solver_options = {'step_size': 0.1}

        # Max memory usage settings
        self.max_integration_buffer = 100  # Maximum number of integration points to process at once

    def safe_odeint(self, func, z0, t_span, method=None, rtol=None, atol=None, options=None):
        """Safe integration with automatic error handling and recovery."""
        if method is None:
            method = self.solver
        if rtol is None:
            rtol = self.solver_rtol
        if atol is None:
            atol = self.solver_atol
        if options is None:
            options = self.solver_options

        try:
            # Try with specified solver
            solution = odeint(
                func, z0, t_span,
                method=method,
                rtol=rtol,
                atol=atol,
                options=options
            )

            # Check for NaN values
            if torch.isnan(solution).any():
                raise ValueError("NaN values in solution")

            return solution

        except Exception as e:
            # Log the error
            print(f"Error in {method} integration: {str(e)}. Trying alternate solver...")

            # Try RK4 as first fallback (more stable)
            if method != 'rk4':
                try:
                    print("Falling back to RK4 solver...")
                    solution = odeint(
                        func, z0, t_span,
                        method='rk4',
                        options={'step_size': 0.2}
                    )

                    if not torch.isnan(solution).any():
                        return solution
                except Exception as e2:
                    print(f"RK4 fallback failed: {str(e2)}")

            # Final fallback to Euler
            print("Using Euler solver as final fallback...")
            solution = self.euler_integration(func, z0, t_span)
            return solution

    def euler_integration(self, func, z0, t_span):
        """Memory-efficient Euler integration."""
        device = z0.device
        batch_size, num_nodes, state_dim = z0.shape
        solution = []

        # Initialize with initial state
        solution.append(z0)
        current_state = z0

        # Process time steps
        for i in range(1, len(t_span)):
            try:
                # Calculate time step
                dt = t_span[i] - t_span[i-1]

                # Calculate derivative
                derivative = func(t_span[i-1], current_state)

                # Apply Euler step with stability check
                next_state = current_state + derivative * dt

                # Handle numerical issues
                if torch.isnan(next_state).any():
                    print(f"NaN detected at step {i}, applying stabilization...")
                    # Use previous state with small perturbation
                    next_state = current_state + 0.001 * torch.randn_like(current_state)

                solution.append(next_state)
                current_state = next_state

                # Clean up to save memory
                del derivative
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in Euler step {i}: {str(e)}")
                # Emergency fallback
                next_state = current_state
                solution.append(next_state)
                current_state = next_state

        # Stack along time dimension
        return torch.stack(solution)

    def forward(self, x):
        """Forward pass with checkpointing and memory optimization."""
        batch_size, seq_len, num_nodes, channels = x.shape

        # Get initial state from the first time step
        initial_x = x[:, 0, :, :]
        z0 = self.encoder(initial_x)

        # Create integration time points based on current settings
        if self.integration_points is None:
            self.integration_points = max(seq_len // SUBSAMPLE_FACTOR, 10)

        # Create time span
        integration_time = torch.linspace(0, seq_len-1, self.integration_points, device=x.device)

        # Process integration in chunks to save memory
        chunk_size = min(self.max_integration_buffer, self.integration_points)
        full_solution = []

        for i in range(0, self.integration_points, chunk_size):
            end_i = min(i + chunk_size, self.integration_points)
            time_chunk = integration_time[i:end_i]

            # If first chunk, use initial state
            if i == 0:
                initial_state = z0
            else:
                # Otherwise use the last state from previous chunk
                initial_state = solution[-1]

            # Integrate this time chunk
            solution = self.safe_odeint(self.ode_func, initial_state, time_chunk)

            # Save only the states we need
            if i == 0:
                full_solution.append(solution)
            else:
                # Skip first state as it's the same as previous chunk's last state
                full_solution.append(solution[1:])

            # Clean up memory
            torch.cuda.empty_cache()

        # Concatenate solution chunks
        try:
            solution = torch.cat(full_solution, dim=0)
        except Exception as e:
            print(f"Error concatenating solutions: {str(e)}")
            # Emergency fallback - create a simpler solution
            print("Using emergency solution generation...")
            solution = torch.zeros(self.integration_points, batch_size, num_nodes, self.hidden_dim, device=x.device)
            solution[0] = z0
            for i in range(1, self.integration_points):
                # Simple dynamics - previous state with small random update
                solution[i] = solution[i-1] + 0.01 * torch.randn_like(solution[i-1])

        # Interpolate solution to original sequence length
        interp_solution = self.interpolate_solution(solution, integration_time, seq_len)

        # Apply decoder to get predictions
        pred_x_flat = interp_solution.reshape(-1, self.hidden_dim)
        pred_obs_flat = self.decoder(pred_x_flat)
        pred_obs = pred_obs_flat.reshape(seq_len, batch_size, num_nodes, channels)

        # Permute to match expected shape [batch_size, seq_len, num_nodes, channels]
        pred_sequences = pred_obs.permute(1, 0, 2, 3)

        return pred_sequences, interp_solution

    def interpolate_solution(self, solution, integration_time, seq_len):
        """Memory-efficient interpolation."""
        # Create full time points for original sequence
        full_time = torch.arange(seq_len, device=solution.device).float()

        # Get dimensions
        batch_size = solution.shape[1]
        num_nodes = solution.shape[2]
        hidden_dim = solution.shape[3]

        # Initialize output
        interp_solution = torch.zeros(seq_len, batch_size, num_nodes, hidden_dim, device=solution.device)

        # Process in chunks for memory efficiency
        chunk_size = 50  # Process 50 time points at once

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            time_chunk = full_time[i:end_i]

            # For each time point, find nearest integration points
            for j, t in enumerate(time_chunk):
                # Find closest time indices
                idx = torch.searchsorted(integration_time, t)
                if idx == 0:
                    # If before first point, use first point
                    interp_solution[i+j] = solution[0]
                elif idx == len(integration_time):
                    # If after last point, use last point
                    interp_solution[i+j] = solution[-1]
                else:
                    # Linear interpolation between points
                    t_prev = integration_time[idx-1]
                    t_next = integration_time[idx]
                    alpha = (t - t_prev) / (t_next - t_prev)

                    interp_solution[i+j] = (1 - alpha) * solution[idx-1] + alpha * solution[idx]

        return interp_solution


# Function to create simplified adjacency matrix based on electrode coordinates
def create_coordinate_adjacency_matrix(coords, threshold=0.3):
    """
    Create an efficient adjacency matrix based on spatial proximity of electrodes.
    This is faster and more memory efficient than correlation-based approaches.
    """
    print("Creating coordinate-based adjacency matrix...")

    # Extract coordinates
    coords_array = coords.values
    num_nodes = len(coords_array)

    # Initialize sparse matrices
    indices = []
    values = []

    # Calculate connections using a threshold on distances
    print("Computing connections...")
    for i in tqdm(range(num_nodes)):
        for j in range(i+1, num_nodes):  # Skip redundant calculations
            distance = np.sqrt(np.sum((coords_array[i] - coords_array[j]) ** 2))

            # Add edges if distance is below threshold
            if distance < threshold:
                # Add both directions (i→j and j→i)
                indices.append([i, j])
                indices.append([j, i])

                # Calculate weight based on distance (closer = stronger)
                weight = 1.0 - (distance / threshold)
                values.extend([weight, weight])

    # Convert to sparse tensor
    if not indices:
        # Fallback if no connections found
        print("Warning: No connections found. Using fallback adjacency.")
        adj_matrix = torch.eye(num_nodes)
    else:
        indices = torch.tensor(indices).t()
        values = torch.tensor(values, dtype=torch.float32)
        adj_matrix = torch.sparse_coo_tensor(
            indices, values, size=(num_nodes, num_nodes)
        ).to_dense()

    edges = (adj_matrix > 0).sum().item()
    print(f"Created coordinate-based adjacency matrix with {edges} edges")

    # Return as torch tensor
    return adj_matrix.to(device)


# Optimized training function
def train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE,
                use_mixed_precision=USE_MIXED_PRECISION,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
    """
    Memory-efficient training function with mixed precision and gradient accumulation.
    """
    print(f"Starting training with mixed precision: {use_mixed_precision}...")
    device = next(model.parameters()).device

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=use_mixed_precision)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        eps=1e-8
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
        pct_start=0.3
    )

    # Track best model and training history
    best_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0

    history = {
        'epoch_losses': [],
        'validation_losses': [],
        'learning_rates': [],
        'solver_methods': [],
        'integration_points': []
    }

    # Training loop
    for epoch in range(epochs):
        # Training mode
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        # Progress bar for this epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()  # Zero gradients at start of epoch

        for batch_idx, (sequences, actual_lengths) in enumerate(train_pbar):
            try:
                # Free memory before processing batch
                torch.cuda.empty_cache()

                # Move data to device
                sequences = sequences.to(device)
                actual_lengths = actual_lengths.to(device)

                # Forward pass with mixed precision
                with autocast(enabled=use_mixed_precision):
                    # Forward pass
                    predictions, _ = model(sequences)

                    # Create mask for actual sequence lengths
                    batch_size, seq_len, num_nodes, channels = sequences.shape
                    mask = torch.zeros(batch_size, seq_len, device=device)
                    for b in range(batch_size):
                        mask[b, :actual_lengths[b]] = 1

                    # Calculate loss with masking
                    mse_loss = ((predictions - sequences) ** 2 * mask.unsqueeze(-1).unsqueeze(-1)).sum() / \
                              (mask.sum() * num_nodes * channels + 1e-8)

                    # Scale loss by accumulation steps
                    loss = mse_loss / gradient_accumulation_steps

                # Backward pass with scaling
                scaler.scale(loss).backward()

                # Update weights if needed
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Update with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Update learning rate
                    scheduler.step()

                # Update stats
                epoch_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix({
                    'loss': loss.item() * gradient_accumulation_steps,
                    'lr': current_lr
                })

                # Memory cleanup
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during batch {batch_idx}: {str(e)}")
                # Skip this batch and continue
                torch.cuda.empty_cache()
                continue

        # Calculate average loss for the epoch
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
        else:
            avg_loss = float('inf')

        # Validation
        model.eval()
        val_loss = validate_model(model, train_loader, max_batches=3,
                                 use_mixed_precision=use_mixed_precision)

        # Update history
        history['epoch_losses'].append(avg_loss)
        history['validation_losses'].append(val_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        history['solver_methods'].append(model.solver)
        history['integration_points'].append(model.integration_points)

        # Log performance
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            # Store on CPU to save GPU memory
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best model saved! Loss: {best_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"results/checkpoints/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Training complete
    print(f"Training completed with best validation loss: {best_loss:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Loaded best model with loss: {best_loss:.6f}")

        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'best_loss': best_loss
        }, "results/final_model.pt")

    # Plot training history
    plot_training_history(history)

    return model, history


def validate_model(model, data_loader, max_batches=None, use_mixed_precision=USE_MIXED_PRECISION):
    """
    Memory-efficient validation function.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (sequences, actual_lengths) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            try:
                # Move data to device
                sequences = sequences.to(device)
                actual_lengths = actual_lengths.to(device)

                # Forward pass with mixed precision
                with autocast(enabled=use_mixed_precision):
                    predictions, _ = model(sequences)

                    # Calculate masked loss
                    batch_size, seq_len, num_nodes, channels = sequences.shape
                    mask = torch.zeros(batch_size, seq_len, device=device)
                    for b in range(batch_size):
                        mask[b, :actual_lengths[b]] = 1

                    loss = ((predictions - sequences) ** 2 * mask.unsqueeze(-1).unsqueeze(-1)).sum() / \
                          (mask.sum() * num_nodes * channels + 1e-8)

                total_loss += loss.item()
                batch_count += 1

                # Clear cache after each batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during validation batch {batch_idx}: {str(e)}")
                continue

    if batch_count > 0:
        return total_loss / batch_count
    else:
        return float('inf')


def plot_training_history(history):
    """
    Plot the training history metrics.
    """
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch_losses'], label='Training Loss')
    plt.plot(history['validation_losses'], label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.grid(True)

    # Plot learning rate
    plt.subplot(2, 2, 2)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # Plot integration points
    plt.subplot(2, 2, 3)
    plt.plot(history['integration_points'])
    plt.xlabel('Epoch')
    plt.ylabel('Integration Points')
    plt.title('Integration Points')
    plt.grid(True)

    # Plot solver methods
    plt.subplot(2, 2, 4)
    methods = history['solver_methods']
    unique_methods = list(dict.fromkeys(methods))
    method_indices = [unique_methods.index(m) for m in methods]
    plt.plot(method_indices)
    plt.yticks(range(len(unique_methods)), unique_methods)
    plt.xlabel('Epoch')
    plt.title('Solver Methods')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()


def visualize_model(model, data_loader, name_suffix="", max_samples=2):
    """
    Create visualizations of model predictions with memory efficiency.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Process a few samples
        for sample_idx, (sequences, actual_lengths) in enumerate(data_loader):
            if sample_idx >= max_samples:
                break

            # Process one sequence at a time to save memory
            for batch_idx in range(min(2, sequences.shape[0])):
                # Extract this sequence
                sequence = sequences[batch_idx:batch_idx+1].to(device)
                length = actual_lengths[batch_idx:batch_idx+1].to(device)

                # Get predictions
                with autocast(enabled=USE_MIXED_PRECISION):
                    predictions, _ = model(sequence)

                # Convert to numpy
                sequence_np = sequence.cpu().numpy()
                predictions_np = predictions.cpu().numpy()

                # Get actual length
                seq_length = length.item()

                # Plot a few representative nodes
                for node_idx in [0, NUM_NODES//3, 2*NUM_NODES//3]:
                    plt.figure(figsize=(10, 6))

                    # Plot actual vs predicted
                    actual = sequence_np[0, :seq_length, node_idx, 0]
                    pred = predictions_np[0, :seq_length, node_idx, 0]

                    plt.plot(actual, label='Actual', linewidth=2)
                    plt.plot(pred, label='Predicted', linewidth=2, linestyle='--')

                    plt.title(f'Node {node_idx} Dynamics')
                    plt.xlabel('Time Step')
                    plt.ylabel('Activity')
                    plt.legend()
                    plt.grid(True)

                    # Save figure
                    plt.savefig(f'results/visualizations/dynamics_sample{sample_idx}_batch{batch_idx}_node{node_idx}_{name_suffix}.png')
                    plt.close()

                # Clear CUDA cache
                torch.cuda.empty_cache()


def main():
    """Main function with memory-efficient implementation."""
    try:
        print("Starting Optimized Neural ODE model for cortical dynamics...")

        # Check CUDA memory
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            print(f"Initial GPU Memory: {free_memory/(1024**3):.2f}GB free / {total_memory/(1024**3):.2f}GB total")

        # Check for existing checkpoints to resume training
        checkpoints = glob.glob("results/checkpoints/checkpoint_epoch_*.pt")
        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoints. Latest: {sorted(checkpoints)[-1]}")
            resume = input("Resume from latest checkpoint? (y/n): ").lower() == 'y'
        else:
            resume = False

        # Data loading with memory optimization
        print("Loading electrode coordinates...")
        coords_path = kagglehub.dataset_download("arunramponnambalam/electrodes-coordinates")
        coords_file = os.path.join(coords_path, "electrodesdata.csv")
        coords = pd.read_csv(coords_file, usecols=[0, 1, 2], names=['x', 'y', 'z'], header=0)

        # Define valid indices - electrodes to keep
        valid_indices = [i for i in range(len(coords)) if i not in
                       [0, 16, 23, 24, 26, 31, 38, 63, 92, 96, 99, 100, 101, 102,
                        108, 111, 112, 113, 114, 122, 128, 129, 139, 142, 145, 146,
                        147, 148, 170, 177, 192, 193]]
        coords = coords.iloc[valid_indices].reset_index(drop=True)

        # Create adjacency matrix from coordinates (more memory efficient)
        adj_matrix = create_coordinate_adjacency_matrix(coords, threshold=0.3)

        # Load neural data in chunks
        print("Loading neural activity data...")
        data_path = kagglehub.dataset_download("nocopyrights/image-stimulus-timestamp-data")
        data_file = os.path.join(data_path, "merged_data.csv")

        # Load in chunks to save memory
        data_chunks = []
        chunk_size = 10000

        for chunk in tqdm(pd.read_csv(data_file, chunksize=chunk_size), desc="Loading data chunks"):
            # Filter out unwanted activities
            filtered_chunk = chunk[
                ~(chunk['activity'].str.startswith('sscr', na=False) |
                  chunk['activity'].str.startswith('sscr2', na=False))
            ]
            data_chunks.append(filtered_chunk)

            # Check if we have enough data
            if sum(len(chunk) for chunk in data_chunks) >= 50000:
                break

        # Combine chunks
        data = pd.concat(data_chunks, ignore_index=True)
        print(f"Loaded dataset with {len(data)} samples")

        # Initialize dataset with subsampling for memory efficiency
        print("Initializing dataset...")
        dataset = NeuralSignalDataset(
            data, coords,
            max_seq_length=MAX_SEQ_LENGTH,
            max_chunks=MAX_CHUNKS,
            subsample_factor=2  # Process every other sample
        )

        # Free memory
        del data, data_chunks
        gc.collect()
        torch.cuda.empty_cache()

        # Create data loader with smaller batch size
        print("Creating data loader...")
        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # No parallel workers to avoid memory issues
            pin_memory=True,
            drop_last=True
        )

        # Initialize model
        print("Initializing model...")
        model = OptimizedNeuralODE(IN_CHANNELS, HIDDEN_DIM, adj_matrix).to(device)

        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {total_params:,} parameters")

        # Resume from checkpoint if requested
        if resume and checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        # Train model with optimized training function
        model, history = train_model(
            model,
            train_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            use_mixed_precision=USE_MIXED_PRECISION,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )

        print("Training completed!")

        # Generate final visualizations
        print("Generating final visualizations...")
        visualize_model(model, train_loader, "final")

        print("Done!")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Caught exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Caught exception: {str(e)}")