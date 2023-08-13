import torch


class Process:
  def __init__(self, chq_text, chq_max_len, tokenizer):
    self.chq_text = chq_text
    self.chq_max_len = chq_max_len
    self.tokenizer = tokenizer


  def pre_process(self):
    chq_text = str(self.chq_text)
    chq_text = " ".join(chq_text.split())

    inputs = self.tokenizer.batch_encode_plus(
        [chq_text],
        max_length=self.chq_max_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length"
        )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long)
        )

  def post_process(self, generated_ids):
    preds = [
        self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
        ]
    return " ".join(preds)