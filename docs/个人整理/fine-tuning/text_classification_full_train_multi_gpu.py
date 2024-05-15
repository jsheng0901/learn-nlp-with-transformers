from accelerate import Accelerator
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,
                          AdamW, get_scheduler)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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

# Using the Accelerate library, enable distributed training on multiple GPUs or TPUs
# Instantiates an Accelerator object that will look at the environment and initialize the proper distributed setup
# Accelerate handles the device placement for you, so you can remove the lines that put the model on the device below
accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# here we use accelerator to set multi GPU device
# This the main step, wrap those objects in the proper container to make sure
# your distributed training works as intended.
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        # already set device in previous step
        # batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward()
        # use accelerator backward instead of original loss backward
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
