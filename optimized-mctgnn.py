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
import gc
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    
    # Clear cache at start
    torch.cuda.empty_cache()
    
    print("Memory status at start:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Constants - minimal for MIG setup
NUM_NODES = 162
IN_CHANNELS = 1
MAX_SEQ_LENGTH = 5000
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
EPOCHS = 50
HIDDEN_DIM = 32
SUBSAMPLE_FACTOR = 10

# Data parsing function
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

# Simplified Dataset
class NeuralSignalDataset(Dataset):
    def __init__(self, data_df, coords_df, max_seq_length=MAX_SEQ_LENGTH, normalize=True,
                 chunk_min_size=20, max_chunks=50):
        print("Initializing dataset...")
        self.data = data_df
        self.coords = coords_df
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.chunk_min_size = chunk_min_size
        self.max_chunks = max_chunks
        self.valid_indices = [i for i in range(194) if i not in [0, 16, 23, 24, 26, 31, 38, 63, 92, 96, 99, 100, 101, 102, 108, 111, 112, 113, 114, 122, 128, 129, 139, 142, 145, 146, 147, 148, 170, 177, 192, 193]]
        self.process_data()

    def process_data(self):
        """Process the neural data into chunks based on stimulus changes."""
        print("Chunking data based on activity changes...")
        chunked_data = []
        actual_lengths = []
        current_activity = None
        current_chunk = []
        
        total_rows = len(self.data)
        sample_size = min(total_rows, 50000)  # Limit sample size
        
        # Use smaller step to process less data
        step = max(1, total_rows // sample_size)
        processed_count = 0
        skipped_count = 0
        chunk_count = 0
        
        # Process in steps to reduce memory usage
        pbar = tqdm(range(0, total_rows, step), desc="Processing data")
        for idx in pbar:
            try:
                electrode_str = self.data.iloc[idx]['data']
                electrode_values = parse_numeric_string(electrode_str)
                
                if electrode_values is None or len(electrode_values) != 194:
                    skipped_count += 1
                    continue
                
                electrode_values = electrode_values[self.valid_indices]
                activity = self.data.iloc[idx]['activity']
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
                
                # Update progress
                if processed_count % 1000 == 0:
                    pbar.set_postfix({
                        'processed': processed_count,
                        'chunks': chunk_count
                    })
                    
                # Check if we have enough chunks
                if chunk_count >= self.max_chunks:
                    break
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                skipped_count += 1
                continue
        
        # Process the last chunk
        if current_chunk and len(current_chunk) > self.chunk_min_size and len(chunked_data) < self.max_chunks:
            chunked_data.append(np.vstack(current_chunk))
            actual_lengths.append(len(current_chunk))
        
        if not chunked_data:
            raise ValueError("No valid data chunks found")
        
        print(f"Found {processed_count} valid rows, created {len(chunked_data)} chunks")
        
        # Process chunks into dataset
        self.electrode_data = []
        self.actual_lengths = []
        
        for chunk, actual_length in zip(chunked_data, actual_lengths):
            if actual_length > self.max_seq_length:
                self.electrode_data.append(chunk[:self.max_seq_length])
                self.actual_lengths.append(self.max_seq_length)
            else:
                pad_len = self.max_seq_length - actual_length
                padded = np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant')
                self.electrode_data.append(padded)
                self.actual_lengths.append(actual_length)
        
        # Stack data
        print("Stacking data...")
        self.electrode_data = np.stack(self.electrode_data)
        
        # Normalize data
        if self.normalize:
            print("Normalizing data...")
            mean = np.mean(self.electrode_data)
            std = np.std(self.electrode_data) + 1e-8
            self.electrode_data = (self.electrode_data - mean) / std
        
        print(f"Dataset prepared. Shape: {self.electrode_data.shape}")

    def __len__(self):
        return len(self.electrode_data)

    def __getitem__(self, idx):
        sequences = self.electrode_data[idx]
        actual_length = self.actual_lengths[idx]
        return torch.FloatTensor(sequences).unsqueeze(-1), torch.tensor(actual_length, dtype=torch.long)

# Simplified GNN function
class StableGNNFunc(nn.Module):
    def __init__(self, hidden_dim, adj_matrix):
        super(StableGNNFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer('adj_matrix', adj_matrix)
        
        # Simpler network architecture
        self.gnn_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.dm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Model parameters
        self.diffusion_coef = 0.02
        self.decay_coef = 0.03
        self.scale_factor = 1.0

    def forward(self, t, z):
        # Extract states
        batch_size, num_nodes, total_dim = z.shape
        x = z[:, :, :self.hidden_dim]  # Neural activity
        m = z[:, :, self.hidden_dim:]  # Memory component
        
        # Concatenate states
        z_cat = torch.cat([x, m], dim=-1)
        
        # Apply GNN transformation
        transformed = self.gnn_layer(z_cat)
        
        # Normalize output
        transformed = F.normalize(transformed, p=2, dim=-1) * (self.hidden_dim**0.5)
        
        # Apply graph diffusion
        neighbor_influence = torch.matmul(self.adj_matrix.float(), transformed)
        
        # Calculate dynamics
        dx_dt = self.scale_factor * self.dx_head(
            transformed + self.diffusion_coef * neighbor_influence - self.decay_coef * x
        )
        
        dm_dt = self.scale_factor * self.dm_head(
            transformed + self.diffusion_coef * neighbor_influence
        )
        
        # Combine derivatives
        dz_dt = torch.cat([dx_dt, dm_dt], dim=-1)
        
        # Clip gradients for stability
        norm = torch.norm(dz_dt, dim=-1, keepdim=True)
        max_norm = 5.0
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        dz_dt = dz_dt * scale
        
        # Check for NaNs
        if torch.isnan(dz_dt).any():
            dz_dt = torch.nan_to_num(dz_dt)
            
        return dz_dt

# Simple neural ODE implementation
class SimpleNeuralODEGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, adj_matrix):
        super(SimpleNeuralODEGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer('adj_matrix', adj_matrix)
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        
        # ODE function
        self.ode_func = StableGNNFunc(hidden_dim, adj_matrix)
    
    def simple_integration(self, z0, seq_len, dt=1.0):
        """Simple Euler integration for ODE"""
        print("Using simple Euler integration...")
        batch_size, num_nodes, total_dim = z0.shape
        
        # Initialize solution
        solution = torch.zeros(seq_len, batch_size, num_nodes, total_dim, device=z0.device)
        solution[0] = z0
        
        # Step through time
        z = z0
        for t in range(1, seq_len):
            t_tensor = torch.tensor(float(t), device=z.device)
            dz = self.ode_func(t_tensor, z) * dt
            z = z + dz
            solution[t] = z
            
            # Clean up periodically
            if t % 500 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return solution
    
    def forward(self, x):
        batch_size, seq_len, num_nodes, channels = x.shape
        
        # Encode initial state
        initial_x = x[:, 0, :, :]
        z0 = self.encoder(initial_x)
        
        # Integrate ODE
        try:
            # Simple integration
            full_solution = self.simple_integration(z0, seq_len)
            
            return full_solution
            
        except Exception as e:
            logger.error(f"Integration error: {str(e)}")
            # Fallback to simple integration
            full_solution = self.simple_integration(z0, seq_len)
            return full_solution

# Main model
class SimpleTemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, adj_matrix):
        super(SimpleTemporalGNN, self).__init__()
        self.neural_ode_gnn = SimpleNeuralODEGNN(in_channels, hidden_dim, adj_matrix)
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
    
    def forward(self, x):
        # Get ODE solution
        solution = self.neural_ode_gnn(x)
        
        # Extract activity state
        pred_x = solution[:, :, :, :self.neural_ode_gnn.hidden_dim]
        
        # Apply decoder
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        num_nodes = x.shape[2]
        
        # Reshape for decoder
        pred_x_flat = pred_x.view(-1, self.neural_ode_gnn.hidden_dim)
        pred_obs_flat = self.decoder(pred_x_flat)
        pred_obs = pred_obs_flat.view(seq_len, batch_size, num_nodes, -1)
        
        # Permute to match expected shape
        pred_sequences = pred_obs.permute(1, 0, 2, 3)
        
        return pred_sequences, solution

# Function to create adjacency matrix
def create_simple_adjacency_matrix(n_electrodes, threshold=0.5):
    """Create a simple adjacency matrix based on node proximity"""
    print("Creating simple adjacency matrix...")
    
    # Just make a fully connected graph with self-loops removed
    adj_matrix = torch.ones(n_electrodes, n_electrodes, dtype=torch.float32)
    adj_matrix.fill_diagonal_(0)  # Remove self-loops
    
    print(f"Created adjacency matrix with {adj_matrix.sum().item()} edges")
    return adj_matrix

def main():
    try:
        os.makedirs("results", exist_ok=True)
        print("\n=== Starting MCTGNN - Minimal Version ===\n")
        
        # Load coordinates
        print("Loading electrode coordinates...")
        try:
            coords_path = kagglehub.dataset_download("arunramponnambalam/electrodes-coordinates")
            coords_file = os.path.join(coords_path, "electrodesdata.csv")
            coords = pd.read_csv(coords_file, usecols=[0, 1, 2], names=['x', 'y', 'z'], header=0)
            valid_indices = [i for i in range(len(coords)) if i not in [0, 16, 23, 24, 26, 31, 38, 63, 92, 96, 99, 100, 101, 102, 108, 111, 112, 113, 114, 122, 128, 129, 139, 142, 145, 146, 147, 148, 170, 177, 192, 193]]
            coords = coords.iloc[valid_indices].reset_index(drop=True)
            print(f"Loaded coordinates for {len(coords)} electrodes")
        except Exception as e:
            logger.error(f"Error loading coordinates: {str(e)}")
            print("Using dummy coordinates")
            # Create dummy coordinates
            coords = pd.DataFrame({
                'x': np.random.randn(NUM_NODES),
                'y': np.random.randn(NUM_NODES),
                'z': np.random.randn(NUM_NODES)
            })
        
        # Load activity data
        print("Loading neural activity data...")
        try:
            data_path = kagglehub.dataset_download("nocopyrights/image-stimulus-timestamp-data")
            data_file = os.path.join(data_path, "merged_data.csv")
            data = pd.read_csv(data_file)
            print(f"Loaded dataset with {len(data)} samples")
            data = data[~(data['activity'].str.startswith('sscr', na=False) | data['activity'].str.startswith('sscr2', na=False))].reset_index(drop=True)
            print(f"Dataset after filtering: {len(data)} samples")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")
        
        # Create simple adjacency matrix
        adj_matrix = create_simple_adjacency_matrix(NUM_NODES)
        adj_matrix = adj_matrix.to(device)
        
        # Create dataset
        try:
            dataset = NeuralSignalDataset(data, coords, max_chunks=10)  # Limit to 10 chunks for speed
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise RuntimeError(f"Failed to create dataset: {str(e)}")
        
        # Free memory
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create data loader (single process)
        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        # Initialize model
        print("Creating model...")
        model = SimpleTemporalGNN(IN_CHANNELS, HIDDEN_DIM, adj_matrix).to(device)
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train model
        print("\n=== Beginning Training ===\n")
        
        for epoch in range(1):  # Just one epoch for testing
            print(f"Epoch {epoch+1}/{EPOCHS}")
            
            # Training loop
            model.train()
            total_loss = 0
            batch_count = 0
            
            for sequences, actual_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                try:
                    # Move data to device
                    sequences = sequences.to(device)
                    actual_lengths = actual_lengths.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    # Get predictions
                    pred_sequences, _ = model(sequences)
                    
                    # Create mask for actual sequence lengths
                    batch_size, seq_len, _, _ = sequences.shape
                    mask = torch.zeros(batch_size, seq_len, device=device)
                    for b in range(batch_size):
                        mask[b, :actual_lengths[b]] = 1
                    
                    # Compute loss
                    loss = ((pred_sequences - sequences) ** 2 * mask.unsqueeze(-1).unsqueeze(-1)).sum() / (mask.sum() * NUM_NODES * IN_CHANNELS)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Invalid loss detected, skipping batch")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Update stats
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Report progress
                    if batch_count % 5 == 0:
                        print(f"Batch {batch_count}, Loss: {loss.item():.6f}")
                    
                    # Clean up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    print(f"Batch error: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            # Report epoch results
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.6f}")
            else:
                print("No valid batches in this epoch")
            
            # Save model
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(checkpoint, f"results/minimal_model_epoch{epoch+1}.pt")
                print(f"Model saved to results/minimal_model_epoch{epoch+1}.pt")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
        
        print("\n=== Training Complete ===\n")
        
        # Evaluate model
        print("Running quick evaluation...")
        model.eval()
        
        with torch.no_grad():
            # Just evaluate on first batch
            for sequences, actual_lengths in train_loader:
                sequences = sequences.to(device)
                actual_lengths = actual_lengths.to(device)
                
                # Get predictions
                pred_sequences, _ = model(sequences)
                
                # Simple visualization
                sample_idx = 0
                node_idx = 0
                
                # Extract sequence
                actual_length = actual_lengths[sample_idx].item()
                actual_values = sequences[sample_idx, :actual_length, node_idx, 0].cpu().numpy()
                pred_values = pred_sequences[sample_idx, :actual_length, node_idx, 0].cpu().numpy()
                
                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(actual_values, label='Actual')
                plt.plot(pred_values, label='Predicted')
                plt.title(f"Neural Activity: Node {node_idx}")
                plt.xlabel("Time")
                plt.ylabel("Activity")
                plt.legend()
                plt.savefig("results/minimal_evaluation.png")
                plt.close()
                
                print("Evaluation plot saved to results/minimal_evaluation.png")
                break
        
        print("Done!")
        
    except Exception as e:
        logger.error(f"Main function error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set specific torch settings for stability
    torch.set_num_threads(1)  # Limit CPU threads
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Ensure using first GPU
    
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        print(f"Total runtime: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Fatal error in main program: {str(e)}")
