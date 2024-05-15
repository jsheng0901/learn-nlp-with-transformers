import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    # target就是label在翻译任务里面
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# model parameters
model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"

# load dataset and metric
raw_datasets = load_dataset("wmt16", "ro-en")
metric = load_metric("sacrebleu")

# Pre-process data with tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"

# 如果使用的是T5预训练模型的checkpoints，需要对特殊的前缀进行检查。T5使用特殊的前缀来告诉模型具体要做的任务
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""

# 预处理所有数据，返回的结果会自动被缓存，避免下次处理的时候重新计算
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Train model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# 训练参数
batch_size = 16
args = Seq2SeqTrainingArguments(
    "test-translation",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,     # Seq2SeqTrainer会不断保存模型，我们需要告诉它至多保存save_total_limit=3个模型
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
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
