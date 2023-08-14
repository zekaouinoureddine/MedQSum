import argparse
from process import Process
from model import MedQSumModel
from config import MedQSumConfig

import torch



def generate_summary(chq_text, model, chq_max_len, tokenizer, device):
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

  generated_summary = data.post_process(generated_ids)
  return generated_summary


def run_inference(model_checkpoint, chq):
    config = MedQSumConfig(model_checkpoint)

    model = MedQSumModel(model=config.model)
    model.to(config.device)
    model.load_state_dict(
        torch.load(
            config.model_path,
            map_location=torch.device(config.device)
        )
    )

    generated_summary = generate_summary(chq, model, config.chq_max_len, config.tokenizer, config.device)
    return generated_summary



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a summary from input CHQ text.")
    parser.add_argument("--model_checkpoint", type=str, default="t5-base", help="Model checkpoint name.")
    parser.add_argument("--input_chq_text", type=str, default="", help="Input text for summarization.")
    args = parser.parse_args()

    generated_summary = run_inference(args.model_checkpoint, args.input_chq_text)
    print(generated_summary)


