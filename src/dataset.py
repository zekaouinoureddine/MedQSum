from utils import add_prefix

import torch
from torch.utils.data import Dataset


class MedQSumDataset(Dataset):
    def __init__(self, chq, summary, tokenizer, chq_max_len, sum_max_len, use_instruction=False):
        self.chq = chq
        self.summary = summary
        self.tokenizer = tokenizer
        self.chq_max_len = chq_max_len
        self.sum_max_len = sum_max_len
        self.use_instruction = use_instruction

    
    def __len__(self):
        return len(self.chq)
    

    def __getitem__(self, index):

        # Determine whether to use instruction fine-tuning or not
        if self.use_instruction:
            chq = str(add_prefix(self.chq[index]))
            summary = str(add_prefix(self.summary[index]))
        else:
            chq = str(self.chq[index])
            summary = str(self.summary[index])

        chq = " ".join(chq.split())
        summary = " ".join(summary.split())

        chq_inputs = self.tokenizer.batch_encode_plus(
            [chq],
            max_length=self.chq_max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length"
        )

        sum_inputs = self.tokenizer.batch_encode_plus(
            [summary],
            max_length=self.sum_max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length"
        )

        chq_input_ids = chq_inputs["input_ids"]
        chq_attention_mask = chq_inputs["attention_mask"]
        
        sum_input_ids = sum_inputs["input_ids"]
        sum_attention_mask = sum_inputs["attention_mask"]

        labels = [label if label != 0 else -100 for label in sum_input_ids]


        return {
            "chq_ids": torch.tensor(chq_input_ids, dtype=torch.long).squeeze(),
            "chq_mask": torch.tensor(chq_attention_mask, dtype=torch.long).squeeze(),
            "sum_ids": torch.tensor(sum_input_ids, dtype=torch.long).squeeze(),
            "sum_mask": torch.tensor(sum_attention_mask, dtype=torch.long).squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long).squeeze(),
        }
