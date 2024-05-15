import torch

from docs.个人整理.bert.modeling_bert import (BertForPreTraining, BertForSequenceClassification,
                                              BertForTokenClassification,
                                              BertForNextSentencePrediction,
                                              BertForQuestionAnswering)
from tokenization_bert import BertTokenizer
from modeling_bert import BertModel
from transformers import AutoTokenizer

# BertTokenizer test
bt = BertTokenizer.from_pretrained('bert-base-uncased')
print(bt('I like natural language progressing!'))

# -------------------------------------------------------------------
# AutoTokenizer test
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("I like natural language progressing!", return_tensors="pt")
print(inputs)

# -------------------------------------------------------------------
# BertModel test
model = BertModel.from_pretrained("bert-base-uncased")
outputs = model(**inputs)
print(outputs.last_hidden_state)

# -------------------------------------------------------------------
# BertPreTrainModel test
model = BertForPreTraining.from_pretrained("bert-base-uncased")
outputs = model(**inputs)
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits

# -------------------------------------------------------------------
# BertForNextSentencePrediction only test
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')
print(encoding)
outputs = model(**encoding)
logits = outputs.logits
# class 1 indicates sequence B is a random sequence.
print(logits[0, 0] < logits[0, 1])

# -------------------------------------------------------------------
# BertForSequenceClassification test
model = BertForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to the
# sequence, as well as compute the attention masks.
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")
#
paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

# output above [1, 2]，apply softmax在这里
paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# Should be paraphrased
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
#
# # Should not be paraphrased
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")

# -------------------------------------------------------------------
# BertForTokenClassification test
model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

label_list = [
    "O",  # Outside a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",  # Beginning of a person's name right after another person's name
    "I-PER",  # Person's name
    "B-ORG",  # Beginning of an organisation right after another organisation
    "I-ORG",  # Organisation
    "B-LOC",  # Beginning of a location right after another location
    "I-LOC"  # Location
]

sequence = ("Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very "
            "close to the Manhattan Bridge.")

# A bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")
# print(inputs)
# [1, 30, 9] 每个token对应每个label的预测概率值
outputs = model(inputs).logits
# [1, 30] 每个token对应每个label的预测最大概率值的index，也就是label的class
predictions = torch.argmax(outputs, dim=2)
for token, prediction in zip(tokens, predictions[0].numpy()):
    print((token, model.config.id2label[prediction]))

# -------------------------------------------------------------------
# BertForQuestionAnswering test
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = ("🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose "
        "architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and "
        "Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep "
        "interoperability between TensorFlow 2.0 and PyTorch.")

questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]
# 输入是 文本 + 多个问题组成的句子对
for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    # [1, 111] -> [1] 所有 start 概率值的 token里面选一个最大的index作为start index
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    # [1, 111] -> [1] 所有 end 概率值的 token里面选一个最大的index作为 end index
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")
