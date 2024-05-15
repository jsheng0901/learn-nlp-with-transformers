import collections

import numpy as np
from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator, AutoTokenizer


def prepare_train_features(examples):
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 我们使用overflow_to_sample_mapping参数来映射切片片ID到原始ID。
    # 比如有2个examples被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping也对应4片
    # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 重新标注数据
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理
        # 将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 区分question和context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 拿到原始的example 下标.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有答案，则使用CLS所在的位置为答案.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案的character级别Start/end位置.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 找到token级别的index start.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # 找到token级别的index end.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 检测答案是否超出文本长度，超出的话也适用CLS index作为标注.
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 如果不超出则找到答案token的start和end位置。.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature, and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# squad_v2等于True或者False分别代表使用SQUAD v1 或者 SQUAD v2。
# 如果使用的是其他数据集，那么True代表的是：模型可以回答“不可回答”问题，也就是部分问题不给出答案，而False则代表所有问题必须回答。
squad_v2 = False
# model parameters
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# load dataset
datasets = load_dataset("squad_v2" if squad_v2 else "squad")

# Pre-process data with tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# 有时候question拼接context，而有时候是context拼接question，不同的模型有不同的要求，因此需要使用padding_side参数来指定。
# 这里数据集对应的是context在右边
pad_on_right = tokenizer.padding_side == "right"
# 一般来说预训练模型输入有最大长度要求，所以我们通常将超长的输入进行截断。但是，如果我们将问答数据三元组<question, context, answer>中
# 的超长context截断，那么我们可能丢掉答案（因为我们是从context中抽取出一个小片段作为答案）。为了解决这个问题，我们把超长的输入切片为多个较短
# 的输入，每个输入都要满足模型最大长度输入要求。由于答案可能存在与切片的地方，因此我们需要允许相邻切片之间有交集，代码中通过doc_stride参数控制。
# 输入feature的最大长度，question和context拼接之后
# 2个切片之间的重合token数量。
max_length = 384
doc_stride = 128
# 预处理所有数据，返回的结果会自动被缓存，避免下次处理的时候重新计算
tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
data_collator = default_data_collator

# Train model
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
args = TrainingArguments(
    f"test-squad",
    evaluation_strategy="epoch",
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,  # 训练的论次
    weight_decay=0.01,
)
# trainer 封装了所有训练过程，这里我们没有传入compute_metrics参数，因为我们需要重新写一个计算metric的function，并且有点复杂对于回答问题
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
# 由于训练时间很长，一般一个epoch要2个小时，一定要保存训练完的model
trainer.save_model("test-squad-trained")

# Evaluation model
# 每个feature里的每个token都会有一个logit。预测answer最简单的方法就是选择start的logits里最大的下标最为answer其实位置，
# end的logits里最大下标作为answer的结束位置。但是，如果我们的输入告诉我们找不到答案：比如start的位置比end的位置下标大，
# 或者start和end的位置指向了question。这个时候，简单的方法是我们继续需要选择第2好的预测作为我们的答案了，实在不行看第3好的预测，以此类推。
# 由于上面的方法不太容易找到可行的答案，我们需要思考更合理的方法。我们将start和end的logits相加得到新的打分，
# 然后去看最好的n_best_size个start和end对。从n_best_size个start和end对里推出相应的答案，然后检查答案是否有效，
# 最后将他们按照打分进行排序，选择得分最高的作为答案。
n_best_size = 20
# 和之前一样将prepare_validation_features函数应用到每个验证集合的样本上。
validation_features = datasets["validation"].map(prepare_validation_features,
                                                 batched=True,
                                                 remove_columns=datasets["validation"].column_names
                                                 )
# 获得所有预测结果
raw_predictions = trainer.predict(validation_features)

validation_features.set_format(type=validation_features.format["type"],
                               columns=list(validation_features.features.keys()))
# 当一个token位置对应着question部分时候，prepare_validation_features函数将offset mappings设定为None，
# 所以我们根据offset mapping很容易可以鉴定token是否在context里面啦。我们同样也同时扔掉了特别长的答案。
# 最后一点事情是如何解决无答案的情况（squad_v2=True的时候）。以上的代码都只考虑了context里面的asnwers，
# 所以我们同样需要将无答案的预测得分进行搜集（无答案的预测对应的CLS token的start和end）。如果一个example样本又多个features，
# 那么我们还需要在多个features里预测是不是都无答案。所以无答案的最终得分是所有features的无答案得分最小的那个。
# 只要无答案的最终得分高于其他所有答案的得分，那么该问题就是无答案。
# 把上述所有的情况整合起来变成function


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # 由于一个example可能被切片成多个features，需要一个features和examples的一个映射map，将所有features里的答案全部收集起来。
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


# 将后处理函数应用到原始预测上：
final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)
# 加载评测指标
metric = load_metric("squad_v2" if squad_v2 else "squad")
# 最后我们基于预测和标注对评测指标进行计算。为了合理的比较，我们需要将预测和标注的格式一致。
# 对于squad2来说，评测指标还需要no_answer_probability参数（由于已经无答案直接设置成了空字符串，所以这里直接将这个参数设置为0.0）
if squad_v2:
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
else:
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)
