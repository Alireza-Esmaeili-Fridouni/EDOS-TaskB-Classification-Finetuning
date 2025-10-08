from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          Trainer,
                          default_data_collator)
from peft import (get_peft_model,
                  LoraConfig,
                  TaskType,
                  PeftType,
                  PromptEncoderConfig,
                  PrefixTuningConfig)
import torch
import util
import config
from typing import Literal

class PEFTModelBuilder:
    def __init__(self, model_name:str, token:str="", mode: Literal["qlora", "ptuning", "prefixtuning"] = "qlora"):
        self.model_name = model_name
        self.token = token
        self.mode = mode.lower() 
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        self.model, self.tokenizer = self._build_model()
        
    def _build_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.token,
            quantization_config=quant_config,
            device_map="auto"
            )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        if self.mode == "qlora":
            target_modules = ["q_proj", "v_proj"]
            config_peft = LoraConfig(
                peft_type=PeftType.LORA,
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                target_modules= target_modules,  
                lora_dropout=0.1,  
                bias="none"
                )
        elif self.mode == "ptuning":
            config_peft = PromptEncoderConfig(
                peft_type=PeftType.P_TUNING,
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=25,
                token_dim=model.config.hidden_size,
                num_transformer_submodules=1,
                num_attention_heads=model.config.num_attention_heads,
                num_layers=model.config.num_hidden_layers,
                encoder_reparameterization_type="MLP",
                encoder_hidden_size=model.config.hidden_size,
                encoder_dropout=0.15
                )
        elif self.mode == "prefixtuning":
            config_peft = PrefixTuningConfig(
                peft_type=PeftType.PREFIX_TUNING,
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=25,
                token_dim=model.config.hidden_size,
                num_transformer_submodules=1,
                num_attention_heads=model.config.num_attention_heads,
                num_layers=model.config.num_hidden_layers,
                encoder_hidden_size=model.config.hidden_size
                )       
        peft_model = get_peft_model(model, config_peft)
        return peft_model, tokenizer
    
    def generate_text(self, input_text, max_new_tokens=512):
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        outputs = [out[len(inp):] for inp, out in zip(inputs["input_ids"], outputs)]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def preprocess_with_tokenizer(self, examples, categories):
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        prompt = util.prompt_filler(
            promt_template=config.prompt,
            instruction=config.instruction,
            categories=categories,
            tokenizer=self.tokenizer,
            text=examples['text'],
            label=examples.get('label_type')
            )
        tokens = self.tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
        input_ids = tokens["input_ids"]
        labels = [
            token if token != self.tokenizer.pad_token_id else -100
            for token in input_ids
        ]
        tokens["labels"] = labels
        return tokens  

    def build_tokenized_dataset(self, train_dataset:str, val_dataset:str , test_dataset:str):
        dataset, categories_label, columns  = util.dataset_builder(train_dataset=train_dataset,
                                                                   test_dataset=test_dataset,
                                                                   val_dataset=val_dataset
                                                                   ) 
        tokenized_dataset = dataset.map(lambda x: self.preprocess_with_tokenizer(x, categories_label),
                                        remove_columns=columns
                                        ) 
        return tokenized_dataset
    
        
class GeneralTrainer:
    def __init__(self, dir:str, model, tokenizer, tokenized_dataset, mode: Literal["qlora", "ptuning", "prefixtuning"] = "qlora"):
        self.dir = dir
        self.model = model
        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset
        self.mode = mode
    
    def execute_training(self):
        if self.mode == "qlora":
            lr = 2e-4
        elif self.mode == "ptuning":
            lr = 5e-5
        elif self.mode == "prefixtuning":
            lr = 1e-4
        training_args = TrainingArguments(
        output_dir=self.dir,
        learning_rate=lr,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, 
        warmup_steps=500
        )
        trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=self.tokenized_dataset['train'],
        eval_dataset=self.tokenized_dataset['validation'],
        tokenizer=self.tokenizer,
        data_collator=default_data_collator
        )
        return trainer.train()
        