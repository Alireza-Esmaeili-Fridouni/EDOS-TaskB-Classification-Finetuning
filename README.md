# 🏷️ EDOS-TaskB-Classification-Finetuning

This repository contains the implementation of a **fine-tuning** for **Task B** of the [SemEval 2023 EDOS (Explainable Detection of Online Sexism)](https://aclanthology.org/2023.semeval-1.305/) challenge.  
The goal of Task B is to perform **multi-class text classification** to identify specific types of sexist content in online texts.  

This project fine-tunes the **SmolLM2-135M-Instruct** model using **Parameter-Efficient Fine-Tuning (PEFT)** methods such as **QLoRA**, **P-Tuning**, and **Prefix-Tuning**.

---
## Table of Contents

- [Motivation](#motivation)
- [Features](#features)  
- [Task Overview](#Task-Overview)  
- [Architecture](#architecture)

---

## 🌟 Motivation

Online sexism and offensive content are growing concerns on social media. Detecting and classifying such content accurately helps create safer online environments.  

Training large language models from scratch is costly, so this project uses **Parameter-Efficient Fine-Tuning (PEFT)** with **SmolLM2-135M-Instruct** to adapt a small, instruction-tuned model for **multi-class text classification** efficiently and effectively.

---

## ⚡ Features

- **Multi-class Text Classification:** Classifies online posts into predefined categories of sexist or offensive content.  
- **PEFT Support:** Fine-tuning using **QLoRA**, **P-Tuning**, or **Prefix-Tuning** for efficient adaptation.  
- **Instruction-Tuned Base Model:** Leverages **SmolLM2-135M-Instruct** for better generalization.  
- **Dataset Integration:** Works seamlessly with **EDOS Task B** datasets (`train.csv`, `val.csv`, `test.csv`).  
- **Lightweight & Efficient:** Uses **4-bit quantization** and PEFT to reduce computational cost.  
- **Customizable Prompts:** Easily modify instruction prompts for classification tasks.  

---

## 📘 Task Overview

**EDOS Task B:** Multi-class classification of sexist statements into predefined fine-grained categories.  
This project explores how small instruction-tuned models can be adapted for this task through efficient fine-tuning techniques.

- **Paper:** [SemEval-2023 Task 10: Explainable Detection of Online Sexism (EDOS)](https://aclanthology.org/2023.semeval-1.305/)
- **Original Repository:** [EDOS GitHub](https://github.com/rewire-online/edos/tree/main)

---

## 🏗️ Architecture
| Module | Purpose |
|--------|---------|
| `config.py` | Defines model configuration, dataset paths, and classification prompt templates. |
| `util.py` | Loads datasets, constructs prompts, and prepares data for tokenization. |
| `llm_builder.py` | Loads the SmolLM2 model, applies PEFT (QLoRA, P-Tuning, Prefix-Tuning), and handles tokenization. |
| `main.py` | Executes the training pipeline using Hugging Face `Trainer` and saves the fine-tuned model. |
| `train.csv / val.csv / test.csv` | Input datasets for Task B classification, containing text and label columns. |

---
