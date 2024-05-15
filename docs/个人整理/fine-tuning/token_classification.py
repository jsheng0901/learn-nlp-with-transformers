import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        # word_ids将每一个 subtokens 位置都对应了一个word的下标。比如第1个位置对应第0个word，然后第2、3个位置对应第1个word。
        # 特殊字符对应了None
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100, so they are automatically
            # ignored in the loss function.
            # 我们通常将特殊字符的label设置为 - 100，在模型中 - 100 通常会被忽略掉不计算loss。
            # 我们有两种对齐label的方式：
            # 多个 subtokens 对齐一个word，对齐一个label
            # 多个 subtokens 的第一个 subtoken 对齐word，对齐一个label，其他 subtokens 直接赋予 - 100.
            # 通过label_all_tokens = True切换。
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    # align 之后的 label 和 input ids 的长度应该是一样的，为了后续的计算每个token的loss
    return tokenized_inputs


def compute_metrics(eval_p):
    predictions, labels = eval_p
    # 对于 token classification 输出是 [batch_size, sequence_length, num_label]
    # 我们需要对最后一个axis进行取最大值找到每个token预测的label
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# choose one task
# 需要是"ner", "pos" 或者 "chunk"
task = "ner"
# model parameters
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# load dataset and label and metric with specific task
datasets = load_dataset("conll2003")
label_list = datasets["train"].features[f"{task}_tags"].feature.names
# metric是计算的方式，比如F1之类的，compute_metrics包含对原始输出数据的处理和调用metric计算结果
metric = load_metric("seqeval")

# Pre-process data with tokenizer
# specific token type according to model name
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# 对 subtokens 对齐的方式
# 由于transformer预训练模型在预训练的时候通常使用的是subword，如果我们的文本输入已经被切分成了word，
# 那么这些word还会被我们的tokenizer继续切分，标注数据通常是在word级别进行标注的，既然word还会被切分成 subtokens，
# 那么意味着我们还需要对标注数据进行subtokens的对齐。同时，由于预训练模型输入格式的要求，往往还需要加上一些特殊符号比如： [CLS] 和 [SEP]。
label_all_tokens = True
# 预处理所有数据，返回的结果会自动被缓存，避免下次处理的时候重新计算
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Train model
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)
# trainer 封装了所有训练过程
trainer = Trainer(
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
