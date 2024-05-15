from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,
                          AdamW, get_scheduler)
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import evaluate


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# load dataset
raw_datasets = load_dataset("glue", "mrpc")
# model choose
checkpoint = "bert-base-uncased"

# Pre-process dataset
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Remove the columns corresponding to values the model does not expect (like the sentence1 and sentence2 columns)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# Rename the column label to labels (because the model expects the argument to be named labels)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# Set the format of the datasets, so they return PyTorch tensors instead of lists
tokenized_datasets.set_format("torch")
# Show the current dataset column name
print('---------------------------------------------------')
print('This is current dataset column name')
print(tokenized_datasets["train"].column_names)
# Define dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Train model
# set model with same checkpoint
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# set model optimizer with Adam weight decay regularization
optimizer = AdamW(model.parameters(), lr=5e-5)
# Learning rate scheduler used by default is a linear decay from the maximum value (5e-5) to 0.
# To properly define it, need step number of training steps, which is number of epochs we want to run multiplied by
# the number of training batches (which is the length of our training dataloader)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# show how many total steps we need train
print('---------------------------------------------------')
print('This is total number of training steps we need')
print(num_training_steps)
# set training device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print('---------------------------------------------------')
print('This is the device we used to train model')
print(device)
# full loop training
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # training by batch on each epoch
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluate model
# We’ve already seen the metric.compute() method, like load_metric method inside trainer,
# but metrics can actually accumulate batches for us as we go over the prediction loop with the method add_batch().
# Once we have accumulated all the batches, we can get the final result with metric.compute().
# Here’s how to implement all of this in an evaluation loop:
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    # evaluate by each batch
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    # get label according to max value index
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()








