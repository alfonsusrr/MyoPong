import argparse
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from modules.training.trainer import TableTennisTrainer

load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified PPO Trainer for Table Tennis")
    
    # Core Environment Arguments
    parser.add_argument("--env-id", type=str, default="myoChallengeTableTennisP1-v0", help="Gymnasium environment ID")
    parser.add_argument("--difficulty", type=int, default=1, help="Curriculum difficulty level (0-4)")
    parser.add_argument("--num-envs", type=int, default=12, help="Number of parallel training environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Training Loop Arguments
    parser.add_argument("--total-timesteps", type=int, default=20000000, help="Total PPO training timesteps")
    parser.add_argument("--n-steps", type=int, default=4096, help="Steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=2048, help="Size of the batch for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.0001, help="Entropy coefficient")
    
    # Feature Toggles
    parser.add_argument("--use-sarl", action="store_true", help="Use Synergy Action Reformulation (SARL)")
    parser.add_argument("--use-hierarchical", action="store_true", help="Use Hierarchical wrapper (Observation Augmentation)")
    parser.add_argument("--use-lattice", action="store_true", help="Use LatticeActorCriticPolicy for exploration")
    parser.add_argument("--use-lstm", action="store_true", help="Use RecurrentPPO (LSTM)")
    
    # SARL Specific
    parser.add_argument("--sar-dir", type=str, default="SAR", help="Directory containing SAR artifacts")
    
    # Hierarchical Specific
    parser.add_argument("--update-freq", type=int, default=10, help="Goal update frequency in Hierarchical wrapper")
    
    # Lattice Specific
    parser.add_argument("--lattice-alpha", type=float, default=1.0, help="Alpha parameter for Lattice noise")
    parser.add_argument("--std-clip-min", type=float, default=1e-3, help="Min std for Lattice")
    parser.add_argument("--std-clip-max", type=float, default=10.0, help="Max std for Lattice")
    parser.add_argument("--std-reg", type=float, default=0.0, help="Std regularization for Lattice")
    
    # LSTM Specific
    parser.add_argument("--lstm-hidden-size", type=int, default=256, help="LSTM hidden state size")
    parser.add_argument("--n-lstm-layers", type=int, default=1, help="Number of stacked LSTM layers")
    
    # Normalization Arguments
    parser.add_argument("--norm-obs", action="store_true", default=True, help="Normalize observations")
    parser.add_argument("--norm-reward", action="store_true", default=True, help="Normalize rewards")
    parser.add_argument("--clip-obs", type=float, default=10.0, help="Clipping value for observations")
    parser.add_argument("--clip-reward", type=float, default=10.0, help="Clipping value for rewards")
    
    # Logging & Callbacks
    parser.add_argument("--log-dir", type=str, default=os.path.join("runs", "unified_ppo"), help="Log directory")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="Optional tensorboard log directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--checkpoint-freq", type=int, default=1000000, help="How many steps between checkpoints")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score improvement threshold to save a model checkpoint (0.0 to disable)")
    parser.add_argument("--eval-freq", type=int, default=100000, help="Evaluate policy every N steps")
    parser.add_argument("--eval-envs", type=int, default=4, help="Number of parallel eval envs")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Total eval episodes per evaluation run")
    parser.add_argument("--render-steps", type=int, default=0, help="Record video every N steps (0 to disable)")
    parser.add_argument("--rollout-steps", type=int, default=500, help="Steps per saved rollout")
    
    # Miscellaneous
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Base policy (MlpPolicy/MlpLstmPolicy)")
    parser.add_argument("--activation-fn", type=str, default="silu", choices=["tanh", "relu", "silu", "elu"], help="Activation function for the policy")
    parser.add_argument("--log-std-init", type=float, default=-0.5, help="Initial log standard deviation")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save the final model")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validation logic
    if args.use_lstm and args.use_lattice:
        raise ValueError("LatticeActorCriticPolicy is not compatible with RecurrentPPO (LSTM).")
    
    # Set default wandb project name based on setup if not provided
    if args.wandb_project is None:
        project_parts = ["myosuite", "ppo"]
        if args.use_lstm:
            project_parts.append("lstm")
        if args.use_hierarchical:
            project_parts.append("hierarchical")
        if args.use_sarl:
            project_parts.append("sarl")
        if args.use_lattice:
            project_parts.append("lattice")
        args.wandb_project = "-".join(project_parts)
        
    trainer = TableTennisTrainer(args)
    trainer.setup_envs()
    trainer.setup_model()
    trainer.setup_callbacks()
    trainer.train()

if __name__ == "__main__":
    main()

