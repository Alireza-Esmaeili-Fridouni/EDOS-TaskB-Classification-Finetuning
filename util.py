import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Value

# Import csv dataset
def dataset_builder(train_dataset, test_dataset, val_dataset):
    train_df = pd.read_csv(train_dataset)
    test_df = pd.read_csv(test_dataset)
    val_df = pd.read_csv(val_dataset)
    categories_label = list(set(train_df['label_type']))
    dataset_train = Dataset.from_pandas(train_df)
    dataset_test = Dataset.from_pandas(test_df)
    dataset_val = Dataset.from_pandas(val_df)
    dataset = DatasetDict({'train': dataset_train, 'validation': dataset_val, 'test': dataset_test})
    return dataset, categories_label, train_df.columns.tolist()

def prompt_filler(prompt_template, instruction, categories, text, tokenizer, label=None):
    prompt = prompt_template.format(categories=categories, text=text)
    txt = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]
    if label:
        txt.append({"role": "assistant", "content": label})
    message = tokenizer.apply_chat_template(
           txt,
           tokenize = False,
           add_generation_prompt=True 
        )
    return message  
        