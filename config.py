instruction = "You are a text classification assistant. Your task is to classify the following text into one of the predefined categories."

prompt = """
Classify the following academic abstract into **exactly one** of the following categories.
Your answer must be **only one of the following labels**, spelled **exactly as shown** — no explanations, no extra words, and no made-up categories.
Categories: {categories}
Instructions:
1. Read the provided text carefully.
2. Select the single category that best describes the text.
3. Your output must be only the name of the category, with no additional text or explanation.
4. If the text does not fit exactly, pick the **closest matching** category from the list.
5. If the text fits into more than one, choose the most relevant one.
6. Do not invent new labels. Do not return anything outside the list.
inputtext: {text}
Category:
"""

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
train_dataset = "data/train.csv"
test_dataset = "data/test.csv"
val_dataset = "data/val.csv"