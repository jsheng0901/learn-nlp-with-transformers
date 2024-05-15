import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def preprocess_function(examples):
    # preprocess by batch
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


def compute_metrics(eval_pred):
    # calculate metric since model output is logits
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


# all text classification task
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
# each task keys for how to preprocess
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# choose one task
task = "cola"
actual_task = "mnli" if task == "mnli-mm" else task
# model parameters
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# load dataset and metric with specific task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# Pre-process data with tokenizer
# specific token type according to model name
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# 使用map函数，将预处理函数prepare_train_features应用到（map) 所有样本上
# 更好的是，返回的结果会自动被缓存，避免下次处理的时候重新计算（但是也要注意，如果输入有改动，可能会被缓存影响！）
encoded_dataset = dataset.map(preprocess_function, batched=True)
# 这里在trainer里面会自动设置为这个padding，padding会把每个batch里面的句子都变成和最长的句子一样的长度
# 所以可以传入trainer也可以不传入
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Train model
# 需要设置label的个数，STS-B是一个回归问题，MNLI是一个3分类问题
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
# 设置 metric 对应的名字
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
# 这是 validation dataset 在task里面的名字
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
# 由于我们微调的任务是文本分类任务，而我们加载的是预训练的语言模型，所以会提示我们加载模型的时候扔掉了一些不匹配的神经网络参数
# 比如：预训练语言模型的神经网络head被扔掉了，同时随机初始化了文本分类的神经网络head
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
# arg设定包含了能够定义训练过程的所有属性
args = TrainingArguments(
    "test-glue",                                # output_dir 存储训练model结果的dir名字
    evaluation_strategy="epoch",                # 每个epoch会做一次验证评估
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,     # 训练用的batch size
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)
# trainer 封装了所有训练过程
trainer = Trainer(
    model,                                          # model 本身
    args,                                           # 训练用到的所有参数
    train_dataset=encoded_dataset["train"],         # train dataset
    eval_dataset=encoded_dataset[validation_key],   # eval dataset
    tokenizer=tokenizer,                            # preprocess tokenizer
    compute_metrics=compute_metrics                 # metric for every eval step to calculate
)
# 开始训练
trainer.train()
