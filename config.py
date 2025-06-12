"""
Configuration file for Connect Four AlphaZero-style AI.
Contains all hyperparameters and settings for training and evaluation.
"""

# Network Architecture
NETWORK_CONFIG = {
    'num_filters': 64,      # Number of convolutional filters
    'num_layers': 4,        # Number of convolutional layers
    'input_channels': 3,    # Input channels (current player, opponent, turn indicator)
    'action_size': 7,       # Number of possible actions (columns)
}

# MCTS Configuration
MCTS_CONFIG = {
    'num_simulations': 800,     # Number of MCTS simulations per move
    'c_puct': 1.0,             # UCB exploration constant
    'temperature': 1.0,         # Temperature for action selection
    'temperature_threshold': 15, # Move number after which temperature becomes 0
}

# Training Configuration
TRAINING_CONFIG = {
    # Network training
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'training_steps': 100,      # Training steps per iteration
    
    # Self-play
    'self_play_games': 100,     # Games per iteration
    'replay_buffer_size': 100000,
    
    # Training schedule
    'num_iterations': 50,       # Total training iterations
    'save_interval': 10,        # Save model every N iterations
    'save_buffer_interval': 20, # Save replay buffer every N iterations
    'plot_interval': 5,         # Plot progress every N iterations
    'eval_interval': 10,        # Evaluate model every N iterations
}

# Evaluation Configuration
EVAL_CONFIG = {
    'num_games_vs_random': 100,     # Games to play against random for evaluation
    'num_games_vs_model': 50,       # Games to play against other models
    'mcts_simulations_eval': 800,   # MCTS simulations during evaluation
    'tournament_games': 50,         # Games per matchup in tournaments
}

# File Paths
PATHS = {
    'models_dir': 'models',
    'data_dir': 'data',
    'plots_dir': 'plots',
    'replay_buffer': 'data/replay_buffer.pkl',
    'best_model': 'models/best_model.pth',
    'latest_model': 'models/latest_model.pth',
}

# Quick Training Configurations for Different Scenarios

def get_quick_config():
    """Quick training configuration for testing (small scale)."""
    config = TRAINING_CONFIG.copy()
    config.update({
        'num_iterations': 5,
        'self_play_games': 10,
        'training_steps': 20,
        'mcts_simulations': 100,
        'save_interval': 2,
        'plot_interval': 2,
    })
    return config

def get_development_config():
    """Development configuration (medium scale for testing)."""
    config = TRAINING_CONFIG.copy()
    config.update({
        'num_iterations': 20,
        'self_play_games': 50,
        'training_steps': 50,
        'mcts_simulations': 400,
        'save_interval': 5,
        'plot_interval': 3,
    })
    return config

def get_production_config():
    """Production configuration (full scale training)."""
    config = TRAINING_CONFIG.copy()
    config.update({
        'num_iterations': 100,
        'self_play_games': 200,
        'training_steps': 200,
        'mcts_simulations': 1200,
        'save_interval': 10,
        'plot_interval': 5,
        'replay_buffer_size': 200000,
    })
    return config

def get_lightweight_config():
    """Lightweight configuration for limited computational resources."""
    config = TRAINING_CONFIG.copy()
    config.update({
        'num_iterations': 30,
        'self_play_games': 25,
        'training_steps': 30,
        'mcts_simulations': 200,
        'save_interval': 5,
        'plot_interval': 3,
        'batch_size': 16,
    })
    
    # Also reduce network size
    network_config = NETWORK_CONFIG.copy()
    network_config.update({
        'num_filters': 32,
        'num_layers': 3,
    })
    
    return config, network_config

# Training Presets
TRAINING_PRESETS = {
    'quick': get_quick_config,
    'development': get_development_config,
    'production': get_production_config,
    'lightweight': get_lightweight_config,
}

def create_custom_config(
    num_iterations=50,
    self_play_games=100,
    mcts_simulations=800,
    training_steps=100,
    batch_size=32,
    learning_rate=0.001,
    network_filters=64,
    network_layers=4,
    **kwargs
):
    """Create a custom configuration with specified parameters."""
    
    # Base configuration
    config = TRAINING_CONFIG.copy()
    
    # Update with provided parameters
    config.update({
        'num_iterations': num_iterations,
        'self_play_games': self_play_games,
        'training_steps': training_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    })
    
    # Update MCTS config
    mcts_config = MCTS_CONFIG.copy()
    mcts_config['num_simulations'] = mcts_simulations
    
    # Update network config
    network_config = NETWORK_CONFIG.copy()
    network_config.update({
        'num_filters': network_filters,
        'num_layers': network_layers,
    })
    
    # Apply any additional kwargs
    config.update(kwargs)
    
    return config, mcts_config, network_config

# Hardware-specific configurations
def get_gpu_config():
    """Configuration optimized for GPU training."""
    config = get_production_config()
    config.update({
        'batch_size': 64,      # Larger batches for GPU
        'training_steps': 300,  # More training steps
    })
    return config

def get_cpu_config():
    """Configuration optimized for CPU training."""
    config = get_lightweight_config()[0]  # Get just the config, not the tuple
    config.update({
        'batch_size': 16,      # Smaller batches for CPU
        'mcts_simulations': 100, # Fewer simulations
        'training_steps': 20,   # Fewer training steps
    })
    return config

# Validation function
def validate_config(config):
    """Validate that configuration parameters are reasonable."""
    errors = []
    
    # Check required fields
    required_fields = ['num_iterations', 'self_play_games', 'training_steps', 'batch_size']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check value ranges
    if config.get('batch_size', 0) <= 0:
        errors.append("batch_size must be positive")
    
    if config.get('learning_rate', 0) <= 0 or config.get('learning_rate', 1) > 1:
        errors.append("learning_rate must be between 0 and 1")
    
    if config.get('num_iterations', 0) <= 0:
        errors.append("num_iterations must be positive")
    
    if config.get('self_play_games', 0) <= 0:
        errors.append("self_play_games must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Default configuration for easy import
DEFAULT_CONFIG = TRAINING_CONFIG
DEFAULT_MCTS_CONFIG = MCTS_CONFIG
DEFAULT_NETWORK_CONFIG = NETWORK_CONFIG
DEFAULT_EVAL_CONFIG = EVAL_CONFIG
