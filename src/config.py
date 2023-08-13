import torch

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

from transformers import  T5ForConditionalGeneration, T5Tokenizer
from transformers import  BartForConditionalGeneration, BartTokenizer
from transformers import  PegasusForConditionalGeneration, PegasusTokenizer



GENERATIVE_CONFIGURATION = {
    "do_sample":True,
    "top_p":0.9,
    "top_k":0,
    "sum_max_len":32,
    "temperature":0.7,
    }  


class MedQSumConfig:
    def __init__(self, 
                model_checkpoint="facebook/bart-large-xsum", 
                chq_max_len=384, 
                sum_max_len=32, 
                train_batch_size=4, 
                valid_batch_size=4,
                learning_rate=3e-5, 
                epochs=4):
        
        self.model_checkpoint = model_checkpoint
        self.chq_max_len = chq_max_len
        self.sum_max_len = sum_max_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = f"./output/sum_model.bin"
        
        if model_checkpoint == "facebook/bart-large-xsum":
            self.tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
            self.model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)

        elif model_checkpoint == "t5-base":
            self.tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
            self.model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)

        elif model_checkpoint == "google/flan-t5-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, 
                                                                    load_in_8bit=True, 
                                                                    return_dict=True, 
                                                                    device_map='auto')
            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
                )

            # Prepare int-8 model for training
            model = prepare_model_for_int8_training(model)

            # Add LoRA adaptor
            self.model = get_peft_model(model, lora_config)
            self.model.print_trainable_parameters()
            self.model.config.use_cache = False

        elif model_checkpoint == "google/pegasus-xsum":
            self.tokenizer = PegasusTokenizer.from_pretrained(model_checkpoint)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)

        else:
            raise ValueError(f"Model checkpoint '{model_checkpoint}' not recognized.")



if __name__ == "__main__":
    model_checkpoint = "facebook/bart-large-xsum"  # Choose your desired model checkpoint
    config = MedQSumConfig(model_checkpoint)