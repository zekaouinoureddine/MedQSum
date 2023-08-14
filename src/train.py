from model import MedQSumModel
from config import MedQSumConfig
from dataset import MedQSumDataset
from utils import get_rouge_scores
from engine import eval_fn, train_fn

import argparse

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import DataLoader



def run(train_data_path, test_data_path, train_bach_size, valid_bach_size, device, lr, epochs, model_checkpoint, model_path):
    
    """
    Train and validate the medical question summarization model.

    Args:
        train_data_path (str): Path to the training data JSON file.
        test_data_path (str): Path to the test data JSON file.
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation.
        device (str): Device to run the training on (e.g., 'cuda' or 'cpu').
        lr (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
        model_checkpoint (str): Pretrained model checkpoint name.
        model_path (str): Path to save the best model checkpoint.

    Returns:
        decoded_preds (list): Decoded predictions from the model.
        decoded_labels (list): Decoded labels for validation data.
    """

    train_df = pd.read_json(train_data_path)[:4]
    test_df = pd.read_json(test_data_path)[:1]

    config = MedQSumConfig(model_checkpoint)

    train_dataset = MedQSumDataset(
        chq=train_df["CHQ"].values,
        summary=train_df["Summary"].values,
        tokenizer=config.tokenizer,
        chq_max_len=config.chq_max_len, 
        sum_max_len=config.sum_max_len
    )

    valid_dataset = MedQSumDataset(
        chq=test_df["CHQ"].values,
        summary=test_df["Summary"].values,
        tokenizer=config.tokenizer,
        chq_max_len=config.chq_max_len, 
        sum_max_len=config.sum_max_len
    )

    train_dataloader = DataLoader(train_dataset,
                                batch_size=train_bach_size,
                                shuffle=True,
                                num_workers=0
                                )

    valid_dataloader = DataLoader(valid_dataset,
                                batch_size=valid_bach_size,
                                shuffle=True,
                                num_workers=0)
    

    model = MedQSumModel(model=config.model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    history = defaultdict(list)
    best_loss = np.inf

    for epoch in range(1, epochs+1):
        print("Epoch: ", epoch)
        train_loss = train_fn(train_dataloader, model, optimizer, device)
        decoded_preds, decoded_labels, valid_loss = eval_fn(valid_dataloader, model, device, tokenizer=config.tokenizer)
        
        print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}\n")

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(valid_loss)

        if valid_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss = valid_loss

    return decoded_preds, decoded_labels




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train summarization model")

    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to the validation data JSON file")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--model_checkpoint", type=str, default="t5-base", help="Model HF Checkpoint")
    parser.add_argument("--model_path", type=str, default="./output/sum_model.bin", help="Path to save the model")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # or choose a specific device

    decoded_preds, decoded_labels = run(
        args.train_data_path, 
        args.test_data_path, 
        args.train_batch_size, 
        args.valid_batch_size, 
        device, 
        args.lr, 
        args.epochs, 
        args.model_checkpoint, 
        args.model_path)
    
    rouge_scores = get_rouge_scores(decoded_preds, decoded_labels)
    print(pd.DataFrame(rouge_scores, index=[0]))