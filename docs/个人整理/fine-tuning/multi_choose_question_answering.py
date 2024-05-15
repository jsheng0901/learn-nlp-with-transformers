import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    当传入一个示例的列表时，它会将大列表中的所有输入/注意力掩码等都压平，并传递给tokenizer.pad方法。
    这将返回一个带有大张量的字典(其大小为(batch_size * 4) x seq_length)
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def compute_metrics(eval_predictions):
    # 定义metric计算
    predictions, label_ids = eval_predictions
    # 输出是 [batch, num_choices] 的 logit
    preds = np.argmax(predictions, axis=1)
    # 这里没有用load_metric里面自带的计算function，直接自己计算accuracy
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def preprocess_function(examples):
    # 预处理每个数据，每个数据的文本会重复四次，每次对应拼接上每个选项，形成文本-选项的四对句子的形式
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in
                        enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


# model parameters
model_checkpoint = "bert-base-uncased"
batch_size = 16

# load dataset
datasets = load_dataset("swag", "regular")
# Pre-process data with tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# 每个选项对应在数据集里面的feature名字
ending_names = ["ending0", "ending1", "ending2", "ending3"]
# 预处理所有数据，返回的结果会自动被缓存，避免下次处理的时候重新计算
encoded_datasets = datasets.map(preprocess_function, batched=True)

# Train model
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
args = TrainingArguments(
    "test-swag",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)
# 开始训练
trainer.train()
