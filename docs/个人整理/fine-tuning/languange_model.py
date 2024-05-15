import math

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling)

# This file contains two types of language model question
# First, Causal language modeling, CLM, predict next word index according to input sentence.
# Second, Masked language modeling, MLM, like Bert pretrain

# 因果语言模型（Causal Language Modeling，CLM）
# model parameters
model_checkpoint = "distilgpt2"

# load dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # 拼接所有文本
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 我们将余数对应的部分去掉。但如果模型支持的话，可以添加padding，可以根据需要定制此部件。
    total_length = (total_length // block_size) * block_size
    # 通过max_len进行分割。
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Pre-process data with tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# 预训练模型时所使用的最大长度。最大长度在这里设置为128，以防显存爆炸
# 我们需要将所有文本连接在一起，然后将结果分割成特定block_size的小块
block_size = 128
# 预处理所有数据，返回的结果会自动被缓存，避免下次处理的时候重新计算
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)

# Train model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
# 训练参数，这里很多用的default参数
training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)
# trainer 封装了所有训练过程
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)
# 开始训练
trainer.train()
# evaluate 结果
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# 掩蔽语言模型（Mask Language Modeling，MLM）
# model parameters
model_checkpoint = "distilroberta-base"

# Pre-process data with tokenizer
# 我们可以从新使用CML里面定义好的处理function
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
# 像之前一样，把文本分组在一起，并把它们分成长度为block_size的样本。如果数据集由单独的句子组成，则可以跳过这一步。
lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)
# 在每个阶段，字符总是以相同的方式被掩盖。通过在data_collator中执行这一步，我们可以确保每次检查数据时都以新的方式完成随机掩蔽。
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Train model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# 训练参数
training_args = TrainingArguments(
    "test-mlm",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)
# trainer 封装了所有训练过程
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)
# 开始训练
trainer.train()
# evaluate 结果
# 与CLM目标相比，困惑度要低得多，因为对于MLM目标，我们只需要对隐藏的mask(在这里占总数的15%)进行预测，同时可以访问其余的mask。
# 因此，对于模型来说，这是一项更容易的任务。
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
