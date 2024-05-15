import numpy as np
import nltk
from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    # summary就是label在摘要生成任务里面，这里同机器翻译任务，我们需要额外自己设置target
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# model parameters
model_checkpoint = "t5-small"

# load dataset and metric
raw_datasets = load_dataset("xsum")
metric = load_metric("rouge")

# Pre-process data with tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 1024
max_target_length = 128

# 如果使用的是T5预训练模型的checkpoints，需要对特殊的前缀进行检查。T5使用特殊的前缀来告诉模型具体要做的任务
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

# 预处理所有数据，返回的结果会自动被缓存，避免下次处理的时候重新计算
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Train model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# 训练参数
batch_size = 16
args = Seq2SeqTrainingArguments(
    "test-summarization",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
)
# 数据收集器data collator，将我们处理好的输入喂给模型
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# trainer 封装了所有训练过程
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# 开始训练
trainer.train()

