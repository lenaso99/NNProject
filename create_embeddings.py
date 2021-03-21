from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset


def encode_dataset(train, test):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoded_data_train = tokenizer(train,
    padding=True,
    return_tensors='pt',)
    encoded_data_test = tokenizer(test,
    padding=True,
    return_tensors='pt',)

    input_ids_train = encoded_data_train["input_ids"]
    input_ids_test = encoded_data_test["input_ids"]

    model = BertModel.from_pretrained('bert-base-cased')
    outputs = model(**encoded_data_train)
    return outputs


if __name__ == '__main__':

    dataset = load_dataset("loading_script.py")
    out = encode_dataset(dataset["train"]["word"], dataset["test"]["word"])

    label = encoding_labels(dataset["train"]["tag"])
    out = encode_dataset(dataset["train"]["word"], dataset["test"]["word"])
