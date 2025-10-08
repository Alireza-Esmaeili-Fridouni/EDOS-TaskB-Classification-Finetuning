import llm_builder as llm
import config
import util

token = ""
peft_model = llm.PEFTModelBuilder(model_name=config.model_name, token=token, mode='ptuning')
tokenized_dataset = peft_model.build_tokenized_dataset(
    train_dataset=config.train_dataset,
    test_dataset=config.test_dataset,
    val_dataset=config.val_dataset
    )
trainer = llm.GeneralTrainer(
    dir='ptuning',
    model=peft_model.model,
    tokenizer=peft_model.tokenizer,
    tokenized_dataset=tokenized_dataset,
    mode="ptuning"
    )
trainer.execute_training()
peft_model.model.save_pretrained("SmolLM2-135M-Instruct-finetuned-ptuning")
peft_model.tokenizer.save_pretrained("SmolLM2-135M-Instruct-finetuned-tokenizer-ptuning")