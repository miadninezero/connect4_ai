"""
Main entry point for the Connect Four AlphaZero-style AI.
Provides a command-line interface for training, evaluation, and playing.
"""

import argparse
import os
import sys
from typing import Optional

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAINING_PRESETS, DEFAULT_CONFIG, DEFAULT_NETWORK_CONFIG, 
    validate_config, create_custom_config
)
from ai.trainer import AlphaZeroTrainer, create_training_config
from ai.evaluate import ModelEvaluator, quick_evaluation
from ai.network import ConnectFourNet
from game.board import ConnectFourBoard
from ai.mcts import MCTS, HumanPlayer

def train_model(preset: str = 'development', custom_config: Optional[dict] = None):
    """Train a new model from scratch."""
    print("=== Connect Four AlphaZero Training ===")
    
    if custom_config:
        config = custom_config
    elif preset in TRAINING_PRESETS:
        if preset == 'lightweight':
            config, network_config = TRAINING_PRESETS[preset]()
        else:
            config = TRAINING_PRESETS[preset]()
    else:
        print(f"Unknown preset '{preset}'. Using development config.")
        config = TRAINING_PRESETS['development']()
    
    # Validate configuration
    try:
        validate_config(config)
        print(f"Using configuration preset: {preset}")
        print(f"Training for {config['num_iterations']} iterations")
        print(f"Self-play games per iteration: {config['self_play_games']}")
        print(f"MCTS simulations: {config.get('mcts_simulations', 800)}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    # Create trainer and start training
    trainer = AlphaZeroTrainer(config)
    trainer.train(config['num_iterations'])
    
    print("Training completed!")
    return trainer

def evaluate_model(model_path: str, opponent_path: Optional[str] = None, num_games: int = 100):
    """Evaluate a trained model."""
    print(f"=== Evaluating Model: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    evaluator = ModelEvaluator(model_path)
    
    # Evaluate against random player
    print("\n1. Evaluating against random player...")
    random_results = evaluator.evaluate_vs_random(num_games)
    
    # Evaluate against another model if provided
    if opponent_path:
        if os.path.exists(opponent_path):
            print(f"\n2. Evaluating against {opponent_path}...")
            model_results = evaluator.evaluate_vs_model(opponent_path, num_games)
        else:
            print(f"Opponent model not found: {opponent_path}")
    
    return random_results

def play_vs_human(model_path: str):
    """Play against human player."""
    print("=== Play vs Human ===")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    evaluator = ModelEvaluator(model_path)
    return evaluator.play_vs_human()

def demo_game(model_path: Optional[str] = None, show_thinking: bool = False):
    """Demonstrate a game between AI and random player."""
    print("=== Demo Game ===")
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        network = ConnectFourNet()
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        network.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using random network")
        network = ConnectFourNet()
    
    from ai.self_play import play_exhibition_game
    result = play_exhibition_game(network, None, mcts_simulations=400, verbose=True)
    
    if result > 0:
        print("AI wins!")
    elif result == 0:
        print("Draw!")
    else:
        print("Random player wins!")

def list_models():
    """List available trained models."""
    print("=== Available Models ===")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("No trained models found.")
        return
    
    for i, model_file in enumerate(sorted(model_files), 1):
        model_path = os.path.join(models_dir, model_file)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"{i}. {model_file} ({size_mb:.1f} MB)")

def quick_test():
    """Quick test of the game logic and basic functionality."""
    print("=== Quick Test ===")
    
    # Test game logic
    print("Testing game logic...")
    board = ConnectFourBoard()
    moves = [3, 3, 2, 4, 1, 5, 0]  # Test moves
    
    for move in moves:
        if board.is_valid_move(move):
            board.make_move(move)
            print(f"Move {move}:")
            print(board)
            if board.game_over:
                break
    
    # Test neural network
    print("\nTesting neural network...")
    network = ConnectFourNet()
    state = board.get_state_tensor()
    policy, value = network.predict(state)
    print(f"Policy: {policy}")
    print(f"Value: {value:.3f}")
    
    print("Basic functionality test completed!")

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Connect Four AlphaZero-style AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --preset quick              # Quick training test
  python main.py train --preset development       # Medium-scale training
  python main.py train --preset production        # Full-scale training
  python main.py evaluate models/best_model.pth   # Evaluate a model
  python main.py play models/best_model.pth       # Play against AI
  python main.py demo                              # Watch AI vs random
  python main.py list                              # List available models
  python main.py test                              # Quick functionality test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--preset', default='development', 
                             choices=['quick', 'development', 'production', 'lightweight'],
                             help='Training configuration preset')
    train_parser.add_argument('--iterations', type=int, help='Number of training iterations')
    train_parser.add_argument('--games', type=int, help='Self-play games per iteration')
    train_parser.add_argument('--simulations', type=int, help='MCTS simulations per move')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('model', help='Path to model file')
    eval_parser.add_argument('--opponent', help='Path to opponent model file')
    eval_parser.add_argument('--games', type=int, default=100, help='Number of evaluation games')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against the AI')
    play_parser.add_argument('model', help='Path to model file')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Demo game (AI vs random)')
    demo_parser.add_argument('--model', help='Path to model file (optional)')
    demo_parser.add_argument('--thinking', action='store_true', help='Show AI thinking process')
    
    # List command
    subparsers.add_parser('list', help='List available models')
    
    # Test command
    subparsers.add_parser('test', help='Quick functionality test')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Create custom config if parameters provided
        custom_config = None
        if any([args.iterations, args.games, args.simulations]):
            custom_config = create_training_config(
                num_iterations=args.iterations or 20,
                self_play_games=args.games or 50,
                mcts_simulations=args.simulations or 400
            )
        
        train_model(args.preset, custom_config)
    
    elif args.command == 'evaluate':
        evaluate_model(args.model, args.opponent, args.games)
    
    elif args.command == 'play':
        play_vs_human(args.model)
    
    elif args.command == 'demo':
        demo_game(args.model, args.thinking)
    
    elif args.command == 'list':
        list_models()
    
    elif args.command == 'test':
        quick_test()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
