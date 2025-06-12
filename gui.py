"""
GUI Interface for Connect Four AlphaZero AI
A comprehensive graphical interface providing all training, evaluation, and gameplay features.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import os
import sys
from typing import Optional
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRAINING_PRESETS, validate_config
from ai.trainer import AlphaZeroTrainer, create_training_config
from ai.evaluate import ModelEvaluator
from ai.network import ConnectFourNet
from game.board import ConnectFourBoard
from ai.mcts import MCTS, HumanPlayer

class ConnectFourGUI:
    """Main GUI application for Connect Four AI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four AlphaZero AI")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Queue for thread communication
        self.message_queue = queue.Queue()
        
        # Current game state
        self.current_board = None
        self.ai_player = None
        self.game_in_progress = False
        
        # Training state
        self.training_thread = None
        self.training_in_progress = False
        
        self.create_widgets()
        self.check_queue()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_play_tab()
        self.create_training_tab()
        self.create_evaluation_tab()
        self.create_demo_tab()
        self.create_models_tab()
        
        # Initialize models list after all tabs are created
        self.refresh_models()
        
    def create_play_tab(self):
        """Create the Play vs AI tab."""
        play_frame = ttk.Frame(self.notebook)
        self.notebook.add(play_frame, text="Play vs AI")
        
        # Model selection
        model_frame = ttk.LabelFrame(play_frame, text="Select AI Model")
        model_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.play_model_var = tk.StringVar()
        self.play_model_combo = ttk.Combobox(model_frame, textvariable=self.play_model_var, width=40)
        self.play_model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_play_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(model_frame, text="Refresh Models", command=self.refresh_models).grid(row=0, column=3, padx=5, pady=5)
        
        # Game settings
        settings_frame = ttk.LabelFrame(play_frame, text="Game Settings")
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(settings_frame, text="MCTS Simulations:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.play_simulations_var = tk.StringVar(value="800")
        ttk.Entry(settings_frame, textvariable=self.play_simulations_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="You are Player:").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.player_choice_var = tk.StringVar(value="Player 2 (O)")
        player_combo = ttk.Combobox(settings_frame, textvariable=self.player_choice_var, 
                                   values=["Player 1 (X)", "Player 2 (O)"], state="readonly", width=15)
        player_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Game control buttons
        control_frame = ttk.Frame(play_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="New Game", command=self.start_new_game).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Reset Game", command=self.reset_game).pack(side='left', padx=5)
        
        # Game board
        self.create_game_board(play_frame)
        
        # Game status
        self.game_status_var = tk.StringVar(value="Select a model and click 'New Game' to start")
        ttk.Label(play_frame, textvariable=self.game_status_var, font=('Arial', 10, 'bold')).pack(pady=5)
        
        # Will refresh models after all tabs are created
        
    def create_game_board(self, parent):
        """Create the visual game board."""
        board_frame = ttk.LabelFrame(parent, text="Game Board")
        board_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create board canvas
        self.board_canvas = tk.Canvas(board_frame, width=490, height=420, bg='blue')
        self.board_canvas.pack(pady=10)
        
        # Bind click events
        self.board_canvas.bind("<Button-1>", self.on_board_click)
        
        # Initialize board display
        self.update_board_display()
        
    def create_training_tab(self):
        """Create the Training tab."""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="Training")
        
        # Training presets
        preset_frame = ttk.LabelFrame(train_frame, text="Training Presets")
        preset_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(preset_frame, text="Preset:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.training_preset_var = tk.StringVar(value="development")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.training_preset_var,
                                   values=list(TRAINING_PRESETS.keys()), state="readonly", width=15)
        preset_combo.grid(row=0, column=1, padx=5, pady=5)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        # Custom training parameters
        params_frame = ttk.LabelFrame(train_frame, text="Training Parameters")
        params_frame.pack(fill='x', padx=10, pady=5)
        
        # Create parameter inputs
        params = [
            ("Iterations:", "training_iterations", "20"),
            ("Self-play Games:", "training_games", "50"),
            ("MCTS Simulations:", "training_simulations", "400"),
            ("Training Steps:", "training_steps", "50"),
            ("Batch Size:", "training_batch", "32"),
            ("Learning Rate:", "training_lr", "0.001")
        ]
        
        self.training_vars = {}
        for i, (label, var_name, default) in enumerate(params):
            row = i // 3
            col = (i % 3) * 2
            
            ttk.Label(params_frame, text=label).grid(row=row, column=col, sticky='w', padx=5, pady=5)
            var = tk.StringVar(value=default)
            ttk.Entry(params_frame, textvariable=var, width=10).grid(row=row, column=col+1, padx=5, pady=5)
            self.training_vars[var_name] = var
        
        # Training control
        control_frame = ttk.Frame(train_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.train_button = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side='left', padx=5)
        
        self.stop_train_button = ttk.Button(control_frame, text="Stop Training", 
                                           command=self.stop_training, state='disabled')
        self.stop_train_button.pack(side='left', padx=5)
        
        # Progress bar
        self.training_progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.training_progress.pack(side='left', padx=10, fill='x', expand=True)
        
        # Training log
        log_frame = ttk.LabelFrame(train_frame, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.training_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize preset values
        self.on_preset_change(None)
        
    def create_evaluation_tab(self):
        """Create the Evaluation tab."""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="Evaluation")
        
        # Model selection
        model_frame = ttk.LabelFrame(eval_frame, text="Model Selection")
        model_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(model_frame, text="Model to Evaluate:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.eval_model_var = tk.StringVar()
        self.eval_model_combo = ttk.Combobox(model_frame, textvariable=self.eval_model_var, width=30)
        self.eval_model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_eval_model).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Opponent Model (optional):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.eval_opponent_var = tk.StringVar()
        self.eval_opponent_combo = ttk.Combobox(model_frame, textvariable=self.eval_opponent_var, width=30)
        self.eval_opponent_combo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_opponent_model).grid(row=1, column=2, padx=5, pady=5)
        
        # Evaluation settings
        settings_frame = ttk.LabelFrame(eval_frame, text="Evaluation Settings")
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(settings_frame, text="Number of Games:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.eval_games_var = tk.StringVar(value="100")
        ttk.Entry(settings_frame, textvariable=self.eval_games_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="MCTS Simulations:").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.eval_simulations_var = tk.StringVar(value="800")
        ttk.Entry(settings_frame, textvariable=self.eval_simulations_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Evaluation buttons
        button_frame = ttk.Frame(eval_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Evaluate vs Random", command=self.evaluate_vs_random).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Evaluate vs Model", command=self.evaluate_vs_model).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Quick Test", command=self.quick_test).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(eval_frame, text="Evaluation Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.eval_results = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.eval_results.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_demo_tab(self):
        """Create the Demo tab."""
        demo_frame = ttk.Frame(self.notebook)
        self.notebook.add(demo_frame, text="Demo")
        
        # Demo settings
        settings_frame = ttk.LabelFrame(demo_frame, text="Demo Settings")
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(settings_frame, text="AI Model:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.demo_model_var = tk.StringVar()
        self.demo_model_combo = ttk.Combobox(settings_frame, textvariable=self.demo_model_var, width=30)
        self.demo_model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(settings_frame, text="Browse", command=self.browse_demo_model).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Opponent:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.demo_opponent_var = tk.StringVar(value="Random Player")
        opponent_combo = ttk.Combobox(settings_frame, textvariable=self.demo_opponent_var,
                                     values=["Random Player", "Another Model"], state="readonly", width=15)
        opponent_combo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="MCTS Simulations:").grid(row=1, column=2, sticky='w', padx=5, pady=5)
        self.demo_simulations_var = tk.StringVar(value="400")
        ttk.Entry(settings_frame, textvariable=self.demo_simulations_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Demo controls
        control_frame = ttk.Frame(demo_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Run Demo", command=self.run_demo).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear Log", command=self.clear_demo_log).pack(side='left', padx=5)
        
        # Demo log
        log_frame = ttk.LabelFrame(demo_frame, text="Demo Game Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.demo_log = scrolledtext.ScrolledText(log_frame, height=20, width=80)
        self.demo_log.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_models_tab(self):
        """Create the Models Management tab."""
        models_frame = ttk.Frame(self.notebook)
        self.notebook.add(models_frame, text="Models")
        
        # Models list
        list_frame = ttk.LabelFrame(models_frame, text="Available Models")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for models
        columns = ('Name', 'Size', 'Modified')
        self.models_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.models_tree.heading(col, text=col)
            self.models_tree.column(col, width=150)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=scrollbar.set)
        
        self.models_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Model actions
        actions_frame = ttk.Frame(models_frame)
        actions_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(actions_frame, text="Refresh List", command=self.refresh_models_tree).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Load Model Info", command=self.load_model_info).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Delete Model", command=self.delete_model).pack(side='left', padx=5)
        
        # Model info
        info_frame = ttk.LabelFrame(models_frame, text="Model Information")
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.model_info = scrolledtext.ScrolledText(info_frame, height=8, width=80)
        self.model_info.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize models tree
        self.refresh_models_tree()
        
    def refresh_models(self):
        """Refresh the models list in combo boxes."""
        models_dir = "models"
        models = []
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    models.append(os.path.join(models_dir, file))
        
        # Update all model combo boxes
        for combo in [self.play_model_combo, self.eval_model_combo, 
                     self.eval_opponent_combo, self.demo_model_combo]:
            combo['values'] = models
            if models and not combo.get():
                combo.set(models[0])
                
    def refresh_models_tree(self):
        """Refresh the models tree view."""
        # Clear existing items
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            return
        
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                filepath = os.path.join(models_dir, file)
                size = os.path.getsize(filepath)
                size_mb = f"{size / (1024*1024):.1f} MB"
                
                import time
                modified = time.ctime(os.path.getmtime(filepath))
                
                self.models_tree.insert('', 'end', values=(file, size_mb, modified))
                
    def browse_play_model(self):
        """Browse for play model file."""
        filename = filedialog.askopenfilename(
            title="Select AI Model",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.play_model_var.set(filename)
            
    def browse_eval_model(self):
        """Browse for evaluation model file."""
        filename = filedialog.askopenfilename(
            title="Select Model to Evaluate",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.eval_model_var.set(filename)
            
    def browse_opponent_model(self):
        """Browse for opponent model file."""
        filename = filedialog.askopenfilename(
            title="Select Opponent Model",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.eval_opponent_var.set(filename)
            
    def browse_demo_model(self):
        """Browse for demo model file."""
        filename = filedialog.askopenfilename(
            title="Select Demo Model",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.demo_model_var.set(filename)
            
    def on_preset_change(self, event):
        """Handle training preset change."""
        preset = self.training_preset_var.get()
        if preset in TRAINING_PRESETS:
            if preset == 'lightweight':
                config, _ = TRAINING_PRESETS[preset]()
            else:
                config = TRAINING_PRESETS[preset]()
            
            # Update parameter fields
            self.training_vars['training_iterations'].set(str(config.get('num_iterations', 20)))
            self.training_vars['training_games'].set(str(config.get('self_play_games', 50)))
            self.training_vars['training_simulations'].set(str(config.get('mcts_simulations', 400)))
            self.training_vars['training_steps'].set(str(config.get('training_steps', 50)))
            self.training_vars['training_batch'].set(str(config.get('batch_size', 32)))
            self.training_vars['training_lr'].set(str(config.get('learning_rate', 0.001)))
            
    def start_training(self):
        """Start training in a separate thread."""
        if self.training_in_progress:
            messagebox.showwarning("Training", "Training is already in progress!")
            return
        
        try:
            # Get training parameters
            config = create_training_config(
                num_iterations=int(self.training_vars['training_iterations'].get()),
                self_play_games=int(self.training_vars['training_games'].get()),
                mcts_simulations=int(self.training_vars['training_simulations'].get()),
                training_steps=int(self.training_vars['training_steps'].get()),
                batch_size=int(self.training_vars['training_batch'].get()),
                learning_rate=float(self.training_vars['training_lr'].get())
            )
            
            validate_config(config)
            
            # Update UI
            self.training_in_progress = True
            self.train_button.config(state='disabled')
            self.stop_train_button.config(state='normal')
            self.training_progress.start()
            
            # Clear log
            self.training_log.delete(1.0, tk.END)
            self.log_message("Starting training...")
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self.training_worker,
                args=(config,),
                daemon=True
            )
            self.training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Error starting training: {str(e)}")
            
    def training_worker(self, config):
        """Training worker function running in separate thread."""
        try:
            # Redirect output to GUI
            import io
            import contextlib
            
            class GUIRedirect:
                def __init__(self, queue):
                    self.queue = queue
                    
                def write(self, text):
                    if text.strip():
                        self.queue.put(('log', text.strip()))
                        
                def flush(self):
                    pass
            
            gui_redirect = GUIRedirect(self.message_queue)
            
            with contextlib.redirect_stdout(gui_redirect), contextlib.redirect_stderr(gui_redirect):
                trainer = AlphaZeroTrainer(config)
                trainer.train(config['num_iterations'])
            
            self.message_queue.put(('training_complete', 'Training completed successfully!'))
            
        except Exception as e:
            self.message_queue.put(('training_error', str(e)))
            
    def stop_training(self):
        """Stop training (placeholder - actual implementation would need training loop modification)."""
        messagebox.showinfo("Stop Training", "Training stop requested. Training will complete current iteration.")
        self.training_in_progress = False
        self.train_button.config(state='normal')
        self.stop_train_button.config(state='disabled')
        self.training_progress.stop()
        
    def start_new_game(self):
        """Start a new game against AI."""
        model_path = self.play_model_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid AI model!")
            return
        
        try:
            simulations = int(self.play_simulations_var.get())
            
            # Load AI model
            evaluator = ModelEvaluator(model_path, mcts_simulations=simulations)
            self.ai_player = MCTS(evaluator.network, num_simulations=simulations)
            
            # Initialize game
            self.current_board = ConnectFourBoard()
            self.game_in_progress = True
            
            # Determine player setup
            human_is_player1 = self.player_choice_var.get() == "Player 1 (X)"
            
            self.update_board_display()
            
            if human_is_player1:
                self.game_status_var.set("Your turn! Click a column to make a move.")
            else:
                self.game_status_var.set("AI is thinking...")
                self.root.after(100, self.ai_move)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error starting game: {str(e)}")
            
    def reset_game(self):
        """Reset the current game."""
        self.current_board = None
        self.ai_player = None
        self.game_in_progress = False
        self.update_board_display()
        self.game_status_var.set("Game reset. Click 'New Game' to start.")
        
    def on_board_click(self, event):
        """Handle board click events."""
        if not self.game_in_progress or not self.current_board:
            return
            
        # Calculate column from click position
        col = int(event.x // 70)  # Each column is 70 pixels wide
        
        if col < 0 or col >= 7:
            return
            
        # Check if it's human's turn
        human_is_player1 = self.player_choice_var.get() == "Player 1 (X)"
        human_turn = (human_is_player1 and self.current_board.current_player == 1) or \
                    (not human_is_player1 and self.current_board.current_player == -1)
        
        if not human_turn:
            return
            
        # Make human move
        if self.current_board.is_valid_move(col):
            self.current_board.make_move(col)
            self.update_board_display()
            
            if self.current_board.game_over:
                self.end_game()
            else:
                self.game_status_var.set("AI is thinking...")
                self.root.after(500, self.ai_move)
        else:
            messagebox.showwarning("Invalid Move", "Column is full! Choose another column.")
            
    def ai_move(self):
        """Make AI move."""
        if not self.game_in_progress or not self.current_board or self.current_board.game_over:
            return
            
        try:
            action = self.ai_player.get_best_move(self.current_board, temperature=0.0)
            self.current_board.make_move(action)
            self.update_board_display()
            
            if self.current_board.game_over:
                self.end_game()
            else:
                self.game_status_var.set("Your turn! Click a column to make a move.")
                
        except Exception as e:
            messagebox.showerror("Error", f"AI move error: {str(e)}")
            
    def end_game(self):
        """End the current game and show results."""
        self.game_in_progress = False
        
        if self.current_board.winner == 2:
            result = "Game ended in a draw!"
        elif self.current_board.winner == 1:
            result = "Player 1 (X) wins!"
        else:
            result = "Player 2 (O) wins!"
            
        self.game_status_var.set(result)
        messagebox.showinfo("Game Over", result)
        
    def update_board_display(self):
        """Update the visual board display."""
        self.board_canvas.delete("all")
        
        # Draw grid
        for row in range(6):
            for col in range(7):
                x1, y1 = col * 70, row * 70
                x2, y2 = x1 + 70, y1 + 70
                
                # Draw cell
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill='blue', outline='navy', width=2)
                
                # Draw piece if present
                if self.current_board:
                    piece = self.current_board.board[row, col]
                    if piece != 0:
                        color = 'red' if piece == 1 else 'yellow'
                        self.board_canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill=color, outline='black', width=2)
                        
        # Draw column numbers
        for col in range(7):
            x = col * 70 + 35
            self.board_canvas.create_text(x, -15, text=str(col), fill='black', font=('Arial', 12, 'bold'))
            
    def evaluate_vs_random(self):
        """Evaluate model against random player."""
        model_path = self.eval_model_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model to evaluate!")
            return
        
        self.run_evaluation_thread('random', model_path)
        
    def evaluate_vs_model(self):
        """Evaluate model against another model."""
        model_path = self.eval_model_var.get()
        opponent_path = self.eval_opponent_var.get()
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model to evaluate!")
            return
            
        if not opponent_path or not os.path.exists(opponent_path):
            messagebox.showerror("Error", "Please select a valid opponent model!")
            return
        
        self.run_evaluation_thread('model', model_path, opponent_path)
        
    def run_evaluation_thread(self, eval_type, model_path, opponent_path=None):
        """Run evaluation in separate thread."""
        def worker():
            try:
                games = int(self.eval_games_var.get())
                simulations = int(self.eval_simulations_var.get())
                
                evaluator = ModelEvaluator(model_path, mcts_simulations=simulations)
                
                if eval_type == 'random':
                    results = evaluator.evaluate_vs_random(games)
                    result_text = f"Evaluation vs Random Player ({games} games):\n"
                    result_text += f"Win Rate: {results['win_rate']:.1f}%\n"
                    result_text += f"Draw Rate: {results['draw_rate']:.1f}%\n"
                    result_text += f"Loss Rate: {results['loss_rate']:.1f}%\n"
                else:
                    results = evaluator.evaluate_vs_model(opponent_path, games)
                    result_text = f"Evaluation vs Model ({games} games):\n"
                    result_text += f"Model: {model_path}\n"
                    result_text += f"Opponent: {opponent_path}\n"
                    result_text += f"Win Rate: {results['win_rate']:.1f}%\n"
                    result_text += f"Draw Rate: {results['draw_rate']:.1f}%\n"
                    result_text += f"Loss Rate: {results['loss_rate']:.1f}%\n"
                
                self.message_queue.put(('eval_result', result_text))
                
            except Exception as e:
                self.message_queue.put(('eval_error', str(e)))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def quick_test(self):
        """Run quick functionality test."""
        def worker():
            try:
                from main import quick_test
                import io
                import contextlib
                
                # Capture output
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    quick_test()
                
                result = output.getvalue()
                self.message_queue.put(('eval_result', f"Quick Test Results:\n{result}"))
                
            except Exception as e:
                self.message_queue.put(('eval_error', str(e)))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def run_demo(self):
        """Run demo game."""
        model_path = self.demo_model_var.get()
        
        def worker():
            try:
                simulations = int(self.demo_simulations_var.get())
                
                if model_path and os.path.exists(model_path):
                    network = ConnectFourNet()
                    import torch
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    network.load_state_dict(checkpoint['model_state_dict'])
                    demo_text = f"Demo: AI ({model_path}) vs Random Player\n"
                else:
                    network = ConnectFourNet()
                    demo_text = "Demo: Random AI vs Random Player\n"
                
                demo_text += "=" * 50 + "\n"
                
                from ai.self_play import play_exhibition_game
                
                # Capture game output
                import io
                import contextlib
                
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = play_exhibition_game(network, None, mcts_simulations=simulations, verbose=True)
                
                game_log = output.getvalue()
                
                if result > 0:
                    demo_text += "AI wins!\n"
                elif result == 0:
                    demo_text += "Draw!\n"
                else:
                    demo_text += "Random player wins!\n"
                
                demo_text += "\nGame Log:\n" + game_log
                
                self.message_queue.put(('demo_result', demo_text))
                
            except Exception as e:
                self.message_queue.put(('demo_error', str(e)))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def clear_demo_log(self):
        """Clear the demo log."""
        self.demo_log.delete(1.0, tk.END)
        
    def load_model_info(self):
        """Load information about selected model."""
        selection = self.models_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model from the list.")
            return
        
        item = self.models_tree.item(selection[0])
        model_name = item['values'][0]
        model_path = os.path.join("models", model_name)
        
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            info = f"Model: {model_name}\n"
            info += f"Path: {model_path}\n"
            info += f"Epoch: {checkpoint.get('epoch', 'Unknown')}\n"
            info += f"Loss: {checkpoint.get('loss', 'Unknown')}\n"
            
            # Get model size info
            size = os.path.getsize(model_path)
            info += f"File Size: {size / (1024*1024):.1f} MB\n"
            
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(1.0, info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model info: {str(e)}")
            
    def delete_model(self):
        """Delete selected model."""
        selection = self.models_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to delete.")
            return
        
        item = self.models_tree.item(selection[0])
        model_name = item['values'][0]
        model_path = os.path.join("models", model_name)
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {model_name}?"):
            try:
                os.remove(model_path)
                self.refresh_models_tree()
                self.refresh_models()
                messagebox.showinfo("Success", f"Model {model_name} deleted successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting model: {str(e)}")
                
    def log_message(self, message):
        """Log message to training log."""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
        
    def check_queue(self):
        """Check message queue for updates from worker threads."""
        try:
            while True:
                message_type, data = self.message_queue.get_nowait()
                
                if message_type == 'log':
                    self.log_message(data)
                elif message_type == 'training_complete':
                    self.log_message(data)
                    self.training_in_progress = False
                    self.train_button.config(state='normal')
                    self.stop_train_button.config(state='disabled')
                    self.training_progress.stop()
                    self.refresh_models()
                    messagebox.showinfo("Training Complete", "Training completed successfully!")
                elif message_type == 'training_error':
                    self.log_message(f"Training error: {data}")
                    self.training_in_progress = False
                    self.train_button.config(state='normal')
                    self.stop_train_button.config(state='disabled')
                    self.training_progress.stop()
                    messagebox.showerror("Training Error", f"Training failed: {data}")
                elif message_type == 'eval_result':
                    self.eval_results.insert(tk.END, data + "\n" + "="*50 + "\n")
                    self.eval_results.see(tk.END)
                elif message_type == 'eval_error':
                    messagebox.showerror("Evaluation Error", f"Evaluation failed: {data}")
                elif message_type == 'demo_result':
                    self.demo_log.insert(tk.END, data + "\n" + "="*50 + "\n")
                    self.demo_log.see(tk.END)
                elif message_type == 'demo_error':
                    messagebox.showerror("Demo Error", f"Demo failed: {data}")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)

def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = ConnectFourGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
