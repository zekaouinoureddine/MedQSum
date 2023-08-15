import argparse
import torch
from process import Process
from model import MedQSumModel
from config import MedQSumConfig


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


def run_inference(args):
    config = MedQSumConfig(args.model_checkpoint)
    model = MedQSumModel(model=config.model)

    model.load_state_dict(
        torch.load(
            config.model_path,
            map_location=torch.device(args.device)
        )
    )
    
    model.to(args.device)
    model.eval()

    generated_summary = generate_summary(args.input_chq_text, model, args.chq_max_len, config.tokenizer, args.device)
    return generated_summary



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a summary from input CHQ text.")
    parser.add_argument("--model_checkpoint", type=str, default="t5-base", help="Model checkpoint name.")
    parser.add_argument("--chq_max_len", type=int, default=384, help="CHQ maximum sequence length.")
    parser.add_argument("--input_chq_text", type=str, default="", help="Input text for summarization.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device GPU/CPU")
    args = parser.parse_args()

    generated_summary = run_inference(args)
    print(generated_summary)