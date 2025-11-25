"""
Training and evaluation utilities for CGCNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm




# def collate_fn(data_list):
#     """
#     Custom collate function to properly batch composition features.
    
#     PyG's default batching concatenates all attributes, but we need
#     composition features to be stacked as [batch_size, comp_dim].
#     """
#     batch = Batch.from_data_list(data_list)
    
#     # Stack composition features properly
#     if len(data_list) > 0 and hasattr(data_list[0], 'comp_features'):
#         comp_features = torch.stack([data.comp_features for data in data_list], dim=0)
#         batch.comp_features = comp_features
    
#     return batch



# def collate_fn(data_list):
#     """
#     Custom collate function to properly batch composition features.
    
#     PyG's default batching concatenates all attributes, but we need
#     composition features to be stacked as [batch_size, comp_dim].
    
#     Args:
#         data_list: List of PyG Data objects
    
#     Returns:
#         Batched Data object with properly shaped comp_features
#     """
#     # Use PyG's default batching for graph structure
#     batch = Batch.from_data_list(data_list)
    
#     # Stack composition features properly
#     if len(data_list) > 0 and hasattr(data_list[0], 'comp_features'):
#         comp_features = torch.stack([data.comp_features for data in data_list], dim=0)
#         batch.comp_features = comp_features
    
#     return batch


"""
Training and evaluation utilities for CGCNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm


# NO collate_fn needed! We'll use a different approach.


class Trainer:
    """
    Trainer for CGCNN models.
    
    Handles training loop, validation, checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './checkpoints',
        model_name: str = 'cgcnn',
        criterion: Optional[nn.Module] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: CGCNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: Device for training (cuda/cpu)
            output_dir: Directory for saving checkpoints
            model_name: Name for saved models
            criterion: Loss function (default: MSELoss)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (Mean Squared Error for regression)
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        # Best validation loss for checkpointing
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Output dir: {output_dir}")
    
    def _prepare_batch(self, batch):
        """
        Prepare batch by reshaping comp_features if needed.
        
        This handles the PyG batching quirk where comp_features get concatenated.
        """
        batch = batch.to(self.device)
        
        # Fix comp_features shape if it was concatenated
        if hasattr(batch, 'comp_features') and batch.comp_features.dim() == 1:
            # Reshape from [batch_size * comp_dim] to [batch_size, comp_dim]
            num_graphs = batch.num_graphs
            comp_dim = batch.comp_features.shape[0] // num_graphs
            batch.comp_features = batch.comp_features.view(num_graphs, comp_dim)
        
        return batch
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            (average_loss, average_mae)
        """
        self.model.train()
        
        total_loss = 0
        total_mae = 0
        n_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            batch = self._prepare_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            # loss = self.criterion(predictions, batch.y)
            loss = self.criterion(predictions.squeeze(), batch.y.squeeze())
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            batch_size = batch.y.size(0)
            total_loss += loss.item() * batch_size
            # total_mae += torch.abs(predictions - batch.y).sum().item()
            total_mae += torch.abs(predictions.squeeze() - batch.y.squeeze()).sum().item()
            
            n_samples += batch_size
            
            # Update progress bar
            # pbar.set_postfix({
            #     'loss': f'{loss.item():.4f}',
            #     'mae': f'{torch.abs(predictions - batch.y).mean().item():.4f}'
            # })
            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'mae': f'{torch.abs(predictions.squeeze() - batch.y.squeeze()).mean().item():.4f}'
                            })
        
        avg_loss = total_loss / n_samples
        avg_mae = total_mae / n_samples
        
        return avg_loss, avg_mae
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate on a data loader.
        
        Args:
            loader: Data loader to validate on
        
        Returns:
            (average_loss, average_mae, predictions, targets)
        """
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        n_samples = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = self._prepare_batch(batch)
            
            # Forward pass
            predictions = self.model(batch)
            
            # Compute loss
            # loss = self.criterion(predictions, batch.y)
            loss = self.criterion(predictions.squeeze(), batch.y.squeeze())
            
            # Track metrics
            batch_size = batch.y.size(0)
            total_loss += loss.item() * batch_size
            # total_mae += torch.abs(predictions - batch.y).sum().item()
            total_mae += torch.abs(predictions.squeeze() - batch.y.squeeze()).sum().item()
            n_samples += batch_size
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
        
        avg_loss = total_loss / n_samples
        avg_mae = total_mae / n_samples
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return avg_loss, avg_mae, predictions, targets
    
    # ... rest of the Trainer class stays the same (train, save_checkpoint, load_checkpoint, test methods) ...
    
    # def train(
    #     self,
    #     n_epochs: int,
    #     early_stopping_patience: int = 15,
    #     save_best_only: bool = True,
    #     verbose: bool = True
    # ) -> Dict:
    #     """
    #     Train the model.
        
    #     Args:
    #         n_epochs: Number of epochs to train
    #         early_stopping_patience: Epochs to wait before early stopping
    #         save_best_only: Only save best model (vs saving every epoch)
    #         verbose: Print training progress
        
    #     Returns:
    #         Training history dictionary
    #     """
    #     print(f"\nStarting training for {n_epochs} epochs...")
    #     print("="*70)
        
    #     patience_counter = 0
    #     start_time = time.time()
        
    #     for epoch in range(1, n_epochs + 1):
    #         epoch_start = time.time()
            
    #         # Train
    #         train_loss, train_mae = self.train_epoch()
            
    #         # Validate
    #         val_loss, val_mae, _, _ = self.validate(self.val_loader)
            
    #         # Learning rate scheduling
    #         self.scheduler.step(val_loss)
    #         current_lr = self.optimizer.param_groups[0]['lr']
            
    #         # Update history
    #         self.history['train_loss'].append(train_loss)
    #         self.history['train_mae'].append(train_mae)
    #         self.history['val_loss'].append(val_loss)
    #         self.history['val_mae'].append(val_mae)
    #         self.history['learning_rate'].append(current_lr)
            
    #         # Checkpointing
    #         if val_loss < self.best_val_loss:
    #             self.best_val_loss = val_loss
    #             self.best_epoch = epoch
    #             patience_counter = 0
                
    #             if save_best_only:
    #                 self.save_checkpoint(epoch, is_best=True)
    #         else:
    #             patience_counter += 1
            
    #         # Print progress
    #         if verbose:
    #             epoch_time = time.time() - epoch_start
    #             print(f"Epoch {epoch:3d}/{n_epochs} | "
    #                   f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | "
    #                   f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | "
    #                   f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
    #         # Early stopping
    #         if patience_counter >= early_stopping_patience:
    #             print(f"\nEarly stopping triggered at epoch {epoch}")
    #             print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
    #             break
        
    #     total_time = time.time() - start_time
    #     print("="*70)
    #     print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    #     print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
    #     return self.history

    def train(
        self,
        n_epochs: int,
        early_stopping_patience: int = 15,
        save_best_only: bool = True,
        verbose: bool = True,
        resume_from: Optional[str] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            n_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            save_best_only: Only save best model (vs saving every epoch)
            verbose: Print training progress
            resume_from: Path to checkpoint to resume from (optional)
        
        Returns:
            Training history dictionary
        """
        # Resume from checkpoint if specified
        start_epoch = 1
        patience_counter = 0
        
        if resume_from is not None:
            print(f"\nResuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.history = checkpoint['history']
            
            start_epoch = checkpoint['epoch'] + 1
            patience_counter = checkpoint.get('patience_counter', 0)
            
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Best val loss so far: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            print(f"  Patience counter: {patience_counter}/{early_stopping_patience}")
        
        print(f"\nStarting training for {n_epochs} epochs (from epoch {start_epoch})...")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, n_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae, _, _ = self.validate(self.val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(current_lr)
            
            # Checkpointing
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                improved = True
                
                if save_best_only:
                    self.save_checkpoint(epoch, patience_counter, is_best=True)
            else:
                patience_counter += 1
            
            # Always save latest checkpoint (for resuming)
            if not save_best_only or epoch % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(epoch, patience_counter, is_best=False)
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                status = "✓" if improved else " "
                print(f"{status} Epoch {epoch:3d}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | "
                    f"Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break
        
        total_time = time.time() - start_time
        print("="*70)
        print(f"Training complete! Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
        return self.history
    
    # def save_checkpoint(self, epoch: int, is_best: bool = False):
    #     """
    #     Save model checkpoint.
        
    #     Args:
    #         epoch: Current epoch
    #         is_best: Whether this is the best model so far
    #     """
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #         'best_val_loss': self.best_val_loss,
    #         'history': self.history
    #     }
        
    #     if is_best:
    #         path = self.output_dir / f'{self.model_name}_best.pth'
    #         torch.save(checkpoint, path)
    #         if epoch % 10 == 0:  # Only print every 10 epochs to reduce clutter
    #             print(f"  → Saved best model to {path}")

    def save_checkpoint(self, epoch: int, patience_counter: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            patience_counter: Current early stopping patience counter
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': patience_counter,
            'history': self.history
        }
        
        if is_best:
            path = self.output_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, path)
            if epoch % 10 == 0:  # Only print every 10 epochs to reduce clutter
                print(f"  → Saved best model to {path}")
        else:
            # Save as latest checkpoint (for resuming)
            path = self.output_dir / f'{self.model_name}_latest.pth'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    @torch.no_grad()
    def test(self) -> Dict:
        """
        Evaluate on test set.
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            print("Warning: No test loader provided")
            return {}
        
        print("\nEvaluating on test set...")
        
        test_loss, test_mae, predictions, targets = self.validate(self.test_loader)
        
        # Compute additional metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        results = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return results


# def test_trainer():
#     """Quick test of trainer."""
#     from torch_geometric.data import Data, Batch
#     from mgnn.models.cgcnn import CGCNN
    
#     print("\nTesting Trainer...")
    
#     # Create dummy dataset
#     def create_dummy_data(n_samples=100):
#         data_list = []
#         for _ in range(n_samples):
#             n_nodes = np.random.randint(15, 25)
#             n_edges = n_nodes * 12
            
#             data = Data(
#                 x=torch.randn(n_nodes, 12),
#                 edge_index=torch.randint(0, n_nodes, (2, n_edges)),
#                 edge_attr=torch.randn(n_edges, 1),
#                 comp_features=torch.randn(71),
#                 y=torch.randn(1)
#             )
#             data_list.append(data)
        
#         return data_list
    
#     # Create data loaders
#     train_data = create_dummy_data(100)
#     val_data = create_dummy_data(20)
    
#     # train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
#     # val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
#     train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
#     # Create model
#     model = CGCNN(
#         node_feature_dim=12,
#         edge_feature_dim=1,
#         comp_feature_dim=71,
#         hidden_dim=64,
#         n_conv=2,
#         n_fc=1,
#         dropout=0.1
#     )
    
#     # Create trainer
#     trainer = Trainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         learning_rate=0.001,
#         device='cpu',
#         output_dir='./test_checkpoints',
#         model_name='test_model'
#     )
    
#     # Train for a few epochs
#     print("\nTraining for 3 epochs...")
#     history = trainer.train(n_epochs=3, early_stopping_patience=5, verbose=True)
    
#     print(f"\nHistory keys: {history.keys()}")
#     print(f"Final train loss: {history['train_loss'][-1]:.4f}")
#     print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
#     print("\n✓ Trainer test passed!")

# def test_trainer():
#     """Quick test of trainer."""
#     from torch_geometric.data import Data
#     from mgnn.models.cgcnn import CGCNN
    
#     print("\nTesting Trainer...")
    
#     # Create dummy dataset
#     def create_dummy_data(n_samples=100):
#         data_list = []
#         for _ in range(n_samples):
#             n_nodes = np.random.randint(15, 25)
#             n_edges = n_nodes * 12
            
#             data = Data(
#                 x=torch.randn(n_nodes, 12),
#                 edge_index=torch.randint(0, n_nodes, (2, n_edges)),
#                 edge_attr=torch.randn(n_edges, 1),
#                 comp_features=torch.randn(71),  # Single vector per graph
#                 y=torch.randn(1)
#             )
#             data_list.append(data)
        
#         return data_list
    
#     # Create data loaders WITH custom collate function
#     train_data = create_dummy_data(100)
#     val_data = create_dummy_data(20)
    
#     print(f"Created {len(train_data)} training samples")
#     print(f"Sample comp_features shape: {train_data[0].comp_features.shape}")
    
#     train_loader = DataLoader(
#         train_data, 
#         batch_size=16, 
#         shuffle=True, 
#         collate_fn=collate_fn
#     )
#     val_loader = DataLoader(
#         val_data, 
#         batch_size=16, 
#         shuffle=False, 
#         collate_fn=collate_fn
#     )
    
#     # Test the batching
#     print("\nTesting batch shapes...")
#     for batch in train_loader:
#         print(f"  Batch size: {batch.num_graphs}")
#         print(f"  Node features: {batch.x.shape}")
#         print(f"  Comp features: {batch.comp_features.shape}")
#         print(f"  Expected: [{batch.num_graphs}, 71]")
#         assert batch.comp_features.shape == (batch.num_graphs, 71), \
#             f"Wrong shape! Got {batch.comp_features.shape}, expected [{batch.num_graphs}, 71]"
#         break
    
#     print("  ✓ Batching works correctly!")
    
#     # Create model
#     model = CGCNN(
#         node_feature_dim=12,
#         edge_feature_dim=1,
#         comp_feature_dim=71,
#         hidden_dim=64,
#         n_conv=2,
#         n_fc=1,
#         dropout=0.1
#     )
    
#     # Create trainer
#     trainer = Trainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         learning_rate=0.001,
#         device='cpu',
#         output_dir='./test_checkpoints',
#         model_name='test_model'
#     )
    
#     # Train for a few epochs
#     print("\nTraining for 3 epochs...")
#     history = trainer.train(n_epochs=3, early_stopping_patience=5, verbose=True)
    
#     print(f"\nHistory keys: {history.keys()}")
#     print(f"Final train loss: {history['train_loss'][-1]:.4f}")
#     print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
#     print("\n✓ Trainer test passed!")

# if __name__ == '__main__':
#     test_trainer()


def test_trainer():
    """Quick test of trainer."""
    from torch_geometric.data import Data
    from mgnn.models.cgcnn import CGCNN
    
    print("\nTesting Trainer...")
    
    # Create dummy dataset
    def create_dummy_data(n_samples=100):
        data_list = []
        for _ in range(n_samples):
            n_nodes = np.random.randint(15, 25)
            n_edges = n_nodes * 12
            
            data = Data(
                x=torch.randn(n_nodes, 12),
                edge_index=torch.randint(0, n_nodes, (2, n_edges)),
                edge_attr=torch.randn(n_edges, 1),
                comp_features=torch.randn(71),
                y=torch.randn(1)
            )
            data_list.append(data)
        
        return data_list
    
    # Create data loaders (no collate_fn needed!)
    train_data = create_dummy_data(100)
    val_data = create_dummy_data(20)
    
    print(f"Created {len(train_data)} training samples")
    print(f"Sample comp_features shape: {train_data[0].comp_features.shape}")
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Test the batching with our fix
    print("\nTesting batch preparation...")
    for batch in train_loader:
        print(f"  Before prepare: comp_features shape = {batch.comp_features.shape}")
        
        # Simulate what _prepare_batch does
        if batch.comp_features.dim() == 1:
            num_graphs = batch.num_graphs
            comp_dim = batch.comp_features.shape[0] // num_graphs
            batch.comp_features = batch.comp_features.view(num_graphs, comp_dim)
        
        print(f"  After prepare:  comp_features shape = {batch.comp_features.shape}")
        print(f"  Expected: [{batch.num_graphs}, 71]")
        assert batch.comp_features.shape == (batch.num_graphs, 71), \
            f"Wrong shape! Got {batch.comp_features.shape}"
        break
    
    print("  ✓ Batch preparation works correctly!")
    
    # Create model
    model = CGCNN(
        node_feature_dim=12,
        edge_feature_dim=1,
        comp_feature_dim=71,
        hidden_dim=64,
        n_conv=2,
        n_fc=1,
        dropout=0.1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        device='cpu',
        output_dir='./test_checkpoints',
        model_name='test_model'
    )
    
    # Train for a few epochs
    print("\nTraining for 3 epochs...")
    history = trainer.train(n_epochs=3, early_stopping_patience=5, verbose=True)
    
    print(f"\nHistory keys: {history.keys()}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    print("\n✓ Trainer test passed!")


if __name__ == '__main__':
    test_trainer()


# To run the test, use the following commands:
# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv run python -m mgnn.models.trainer