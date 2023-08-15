from config import GENERATIVE_CONFIGURATION

import torch
from tqdm import tqdm


def train_fn(train_dataloader, model, optimizer, device):
    model.train()
    final_loss = 0

    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        chq_ids = data["chq_ids"].to(device)
        chq_mask = data["chq_mask"].to(device)

        labels = data["labels"].to(device)
        labels_attention_mask = data["sum_mask"].to(device)

        loss, logits = model(
            input_ids=chq_ids,
            attention_mask=chq_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        final_loss += loss.item()
        loss.backward()
        optimizer.zero_grad()
    
    return final_loss/len(train_dataloader)


def eval_fn(valid_dataloader, model, device, tokenizer):
    model.train()
    final_loss = 0
    decoded_labels = []
    decoded_preds = []
    with torch.no_grad():
        for data in tqdm(valid_dataloader, total=len(valid_dataloader)):
            chq_ids = data["chq_ids"].to(device)
            chq_mask = data["chq_mask"].to(device)

            labels = data["labels"].to(device)
            labels_attention_mask = data["sum_mask"].to(device)

            loss, logits = model(
                input_ids=chq_ids,
                attention_mask=chq_mask,
                decoder_attention_mask=labels_attention_mask,
                labels=labels,
            )
            final_loss += loss.item()

            generated_ids = model.model.generate(
                input_ids=chq_ids,
                do_sample=GENERATIVE_CONFIGURATION["do_sample"],
                top_p=GENERATIVE_CONFIGURATION["top_p"],
                top_k=GENERATIVE_CONFIGURATION["top_k"],
                max_length=GENERATIVE_CONFIGURATION["sum_max_len"],
                temperature=GENERATIVE_CONFIGURATION["temperature"],
            )

            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds.extend([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids])
            decoded_labels.extend([tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels])
    
    return decoded_preds, decoded_labels, final_loss/len(valid_dataloader)