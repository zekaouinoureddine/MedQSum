import argparse
from process import Process
from model import MedQSumModel
from config import MedQSumConfig

import torch



def get_important_paragh(chq_text, model, chq_max_len, tokenizer, device):
  data = Process(chq_text, chq_max_len, tokenizer)
  input_ids, attention_mask = data.pre_process()
  input_ids = input_ids.to(device)
  attention_mask = attention_mask.to(device)

  with torch.no_grad():
    generated_ids = model.model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_length=16,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )

  predicted_sum = data.post_process(generated_ids)
  return predicted_sum


def generate_summary(model_checkpoint, chq):
    config = MedQSumConfig(model_checkpoint)

    model = MedQSumModel(model=config.model)
    model.to(config.device)
    model.load_state_dict(
        torch.load(
            config.model_path,
            map_location=torch.device(config.device)
        )
    )

    predicted_sum = get_important_paragh(chq, model, config.chq_max_len, config.tokenizer, config.device)
    print(predicted_sum)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a summary from input CHQ text.")
    parser.add_argument("--model_checkpoint", type=str, default="t5-base", help="Model checkpoint name.")
    parser.add_argument("--input_chq_text", type=str, default="", help="Input text for summarization.")
    args = parser.parse_args()

    generate_summary(args.model_checkpoint, args.input_chq_text)


