from model import MedQSumModel
from config import MedQSumConfig
from dataset import MedQSumDataset
from utils import get_rouge_scores
from engine import eval_fn, train_fn

import wandb
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train summarization model")

    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to the validation data JSON file")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu") , help="Device GPU/CPU")
    parser.add_argument("--chq_max_len", type=int, default=384, help="CHQ maximum sequence length")
    parser.add_argument("--sum_max_len", type=int, default=32, help="Summary maximum sequence length")
    parser.add_argument("--model_checkpoint", type=str, default="t5-base", help="Model HF Checkpoint")
    parser.add_argument("--use_instruction", type=bool, default=False, help="Whether to use instruction fine-tuning or not (True/False)")
    parser.add_argument("--model_path", type=str, default="./output/medqsum_model.bin", help="Path to save the model")

    return parser.parse_args()


def run_medqsum(args):
    """
    Train and validate the medical question summarization model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        decoded_preds (list): Decoded predictions from the model.
        decoded_labels (list): Decoded labels for validation data.
    """

    train_df = pd.read_json(args.train_data_path)[:3]
    valid_df = pd.read_json(args.valid_data_path)[:1]

    config = MedQSumConfig(args.model_checkpoint)

    train_dataset = MedQSumDataset(
        chq=train_df["CHQ"].values,
        summary=train_df["Summary"].values,
        tokenizer=config.tokenizer,
        chq_max_len=args.chq_max_len, 
        sum_max_len=args.sum_max_len,
        use_instruction=args.use_instruction
    )

    valid_dataset = MedQSumDataset(
        chq=valid_df["CHQ"].values,
        summary=valid_df["Summary"].values,
        tokenizer=config.tokenizer,
        chq_max_len=args.chq_max_len, 
        sum_max_len=args.sum_max_len,
        use_instruction=args.use_instruction
    )

    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                num_workers=0
                                )

    valid_dataloader = DataLoader(valid_dataset,
                                batch_size=args.valid_batch_size,
                                shuffle=True,
                                num_workers=0)

    model = MedQSumModel(model=config.model)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_loss = np.inf

    # Initialize WandB and log hyperparameters to WandB
    wandb.init(project="MedQSum", name="MedQSum Model Training", config=vars(args))

    for epoch in range(1, args.epochs+1):
        print("Epoch: ", epoch)
        train_loss = train_fn(train_dataloader, model, optimizer, args.device)
        decoded_preds, decoded_labels, valid_loss = eval_fn(valid_dataloader, model, args.device, tokenizer=config.tokenizer)
        
        print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}\n")

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(valid_loss)

        # Log losses to WandB
        wandb.log({"Train Loss": train_loss, "Valid Loss": valid_loss})

        # Log ROUGE scores
        wandb.log(get_rouge_scores(decoded_preds, decoded_labels))

        if valid_loss < best_loss:
            torch.save(model.state_dict(), args.model_path)
            best_loss = valid_loss

    return decoded_preds, decoded_labels


if __name__ == "__main__":
    args = parse_args()
    decoded_preds, decoded_labels = run_medqsum(args)
    
    rouge_scores = get_rouge_scores(decoded_preds, decoded_labels)
    print(pd.DataFrame(rouge_scores, index=[0]))