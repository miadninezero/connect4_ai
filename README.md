# Connect Four AlphaZero AI

A Connect Four AI that learns purely through self-play using Monte Carlo Tree Search (MCTS) guided by a neural network, inspired by the AlphaZero algorithm. The AI starts from zero knowledge and improves over time to become a strong player.

## 🎯 Features

- **Zero Knowledge Learning**: Starts with no Connect Four knowledge and learns purely through self-play
- **MCTS + Neural Network**: Uses Monte Carlo Tree Search guided by a deep neural network
- **AlphaZero-Style Training**: Combines policy and value networks with MCTS for strong play
- **Progressive Improvement**: AI gets stronger over time through iterative training
- **Human Playable**: Play against the trained AI interactively
- **Comprehensive Evaluation**: Track progress and compare different model versions
- **Flexible Configuration**: Multiple training presets and custom configurations
- **GUI Interface**: Complete graphical user interface for all operations
- **Visual Game Board**: Interactive Connect Four board for human vs AI games

## 🏗 Architecture

### Core Components

- **Game Logic** (`game/board.py`): Complete Connect Four implementation
- **Neural Network** (`ai/network.py`): Policy + Value network using PyTorch
- **MCTS** (`ai/mcts.py`): Monte Carlo Tree Search with neural network guidance
- **Self-Play** (`ai/self_play.py`): Generates training data through AI vs AI games
- **Training** (`ai/trainer.py`): Main training loop coordinating all components
- **Evaluation** (`ai/evaluate.py`): Testing against random players, other models, and humans

### Training Process

1. **Self-Play**: AI plays games against itself using MCTS + current neural network
2. **Data Collection**: Store (state, MCTS policy, game outcome) tuples
3. **Network Training**: Train neural network on collected data
4. **Iteration**: Repeat with improved network, gradually getting stronger

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
   ```bash
   cd connect4_ai
   pip install -r requirements.txt
   ```

2. **Quick Test**:
   ```bash
   python main.py test
   ```

### Training Your First AI

**Quick Training** (5 iterations, ~5 minutes):
```bash
python main.py train --preset quick
```

**Development Training** (20 iterations, ~30 minutes):
```bash
python main.py train --preset development
```

**Production Training** (100 iterations, several hours):
```bash
python main.py train --preset production
```

### Playing Against the AI

```bash
python main.py play models/checkpoint_iter_final.pth
```

### Watching AI vs Random Demo

```bash
python main.py demo --model models/checkpoint_iter_final.pth
```

### Using the GUI Interface

**Launch the GUI**:
```bash
python gui.py
```

The GUI provides a complete interface with 5 main tabs:

1. **Play vs AI**: Interactive game board to play against trained models
2. **Training**: Start and monitor training with real-time logs
3. **Evaluation**: Test models against random players or other models
4. **Demo**: Watch AI vs AI or AI vs Random demonstrations
5. **Models**: Manage saved models, view info, and delete old models

## 📋 Usage Examples

### Training Commands

```bash
# Quick test training
python main.py train --preset quick

# Custom training parameters
python main.py train --iterations 30 --games 75 --simulations 600

# Lightweight training for limited resources
python main.py train --preset lightweight
```

### Evaluation Commands

```bash
# Evaluate model against random player
python main.py evaluate models/checkpoint_iter_20.pth

# Compare two models
python main.py evaluate models/latest.pth --opponent models/older.pth --games 200

# List all available models
python main.py list
```

### Interactive Play

```bash
# Play against the AI (you are Player 2, AI is Player 1)
python main.py play models/best_model.pth

# Watch AI vs random demonstration
python main.py demo --model models/checkpoint_iter_50.pth
```

## ⚙️ Configuration

### Training Presets

- **`quick`**: 5 iterations, 10 games, 100 MCTS simulations (testing)
- **`development`**: 20 iterations, 50 games, 400 simulations (development)
- **`production`**: 100 iterations, 200 games, 1200 simulations (full training)
- **`lightweight`**: 30 iterations, 25 games, 200 simulations (limited resources)

### Custom Configuration

Edit `config.py` or use command-line parameters:

```python
# Example custom configuration
config = {
    'num_iterations': 50,
    'self_play_games': 100,
    'mcts_simulations': 800,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_filters': 64,
    'num_layers': 4
}
```

## 📊 Training Progress

The AI tracks various metrics during training:

- **Loss curves** (policy loss, value loss, total loss)
- **Self-play statistics** (win rates for each player, draw rates)
- **Model evaluation** (performance vs random player over time)
- **Replay buffer size** (amount of training data collected)

All plots are saved to the `plots/` directory.

## 🖥️ GUI Interface Features

### Play vs AI Tab
- **Interactive Game Board**: Visual Connect Four board with click-to-play
- **Model Selection**: Choose from available trained models
- **Game Settings**: Configure MCTS simulations and player roles
- **Real-time Gameplay**: Immediate AI responses and move validation

### Training Tab
- **Training Presets**: Quick, Development, Production, and Lightweight configurations
- **Custom Parameters**: Fine-tune iterations, games, simulations, batch size, and learning rate
- **Real-time Logs**: Live training progress with detailed output
- **Progress Tracking**: Visual progress bar and status updates

### Evaluation Tab
- **Model Testing**: Evaluate against random players or other models
- **Batch Evaluation**: Run multiple games for statistical significance
- **Results Display**: Detailed win/loss/draw statistics
- **Quick Tests**: Functionality verification and basic testing

### Demo Tab
- **AI vs Random**: Watch trained models play against random opponents
- **Game Logging**: Detailed move-by-move game progression
- **Multiple Demos**: Run multiple demonstration games
- **Performance Analysis**: Observe AI decision-making patterns

### Models Tab
- **Model Management**: View all saved models with details
- **Model Information**: File size, training epoch, loss metrics
- **Model Comparison**: Easy selection for evaluations and games
- **Model Cleanup**: Delete old or unwanted model files

## 📁 Project Structure

```
connect4_ai/
│
├── game/
│   ├── __init__.py
│   └── board.py              # Connect Four game logic
│
├── ai/
│   ├── __init__.py
│   ├── network.py            # Neural network (policy + value)
│   ├── mcts.py               # Monte Carlo Tree Search
│   ├── self_play.py          # Self-play data generation
│   ├── trainer.py            # Training orchestration
│   └── evaluate.py           # Model evaluation and testing
│
├── models/                   # Saved model checkpoints
├── data/                     # Training data and replay buffers
├── plots/                    # Training progress plots
│
├── config.py                 # Configuration and hyperparameters
├── main.py                   # Command-line interface
├── gui.py                    # Graphical user interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧠 How It Works

### Neural Network Architecture

- **Input**: 3×6×7 tensor representing board state
  - Channel 0: Current player's pieces
  - Channel 1: Opponent's pieces  
  - Channel 2: Turn indicator
- **Backbone**: Convolutional layers with batch normalization
- **Policy Head**: Outputs probabilities for 7 columns
- **Value Head**: Outputs position evaluation (-1 to +1)

### MCTS Integration

1. **Selection**: Traverse tree using UCB + prior probabilities
2. **Expansion**: Add new node and evaluate with neural network
3. **Simulation**: Use neural network value (no random rollouts)
4. **Backpropagation**: Update visit counts and values up the tree

### Self-Play Training

1. Generate games using current best network
2. Store training examples: (board state, MCTS policy, game result)
3. Train network on replay buffer data
4. Repeat with improved network

## 📈 Expected Performance

As training progresses, you should see:

- **Early iterations** (~5-10): Win rate vs random ~60-70%
- **Mid training** (~20-30): Win rate vs random ~80-90%  
- **Late training** (~50+): Win rate vs random ~95%+

The AI learns fundamental Connect Four concepts:
- Center control in opening
- Threat recognition and blocking
- Building multiple threats
- Endgame technique

## 🔧 Troubleshooting

### Common Issues

**ImportError**: Ensure all dependencies are installed:
```bash
pip install torch numpy tqdm matplotlib
```

**CUDA Issues**: The code works on CPU by default. For GPU:
```python
# In network.py, add device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Memory Issues**: Use smaller configurations:
```bash
python main.py train --preset lightweight
```

**Slow Training**: Reduce MCTS simulations:
```bash
python main.py train --simulations 200
```

### Performance Tips

- **GPU Training**: Significantly faster for larger networks
- **Batch Size**: Increase if you have more memory
- **MCTS Simulations**: More simulations = stronger play but slower training
- **Parallel Self-Play**: Advanced users can modify for parallel game generation

## 🎮 Playing Tips

When playing against the AI:
- Columns are numbered 0-6 (left to right)
- You are Player 2 (O), AI is Player 1 (X)
- Try to build multiple threats simultaneously
- Watch for the AI's tactical patterns and learn from them

## 📚 References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (AlphaZero paper)
- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero paper)
- [A Simple Alpha(Go) Zero Tutorial](https://web.stanford.edu/~surag/posts/alphazero.html)

## 🤝 Contributing

Feel free to improve the code:
- ✅ **GUI Interface**: Complete tkinter-based GUI implemented
- **Parallel Self-Play**: Implement multi-process self-play for faster training
- **Opening Book**: Add Connect Four opening theory integration
- **Network Optimization**: Experiment with different architectures (ResNet, Attention)
- **Board Variations**: Support different board sizes (5x4, 8x7, etc.)
- **Advanced Features**: Tournament mode, ELO ratings, game analysis tools

## 📄 License

This project is open source and available under the MIT License.

---

**Enjoy training your Connect Four AI! 🎉**

The AI will start weak but quickly learn the game through pure self-play. Watch it develop from random moves to sophisticated strategic play!
