import argparse
from classifiers import TextClassifier, LSTM
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel


def train_cnn(model, trainset, word_to_ix, labeldict):

    optimizer = optim.RMSprop(model.parameters(), lr=0.5)

    for epoch in range(5):
      model.train()
      predictions = []
      for (words, tags) in zip(trainset["word"], trainset["tag"]):
         sentence_in = prepare_sequence(words, word_to_ix)
         targets = prepare_sequence(tags, labeldict)
         tags = tags.type(torch.FloatTensor)
         tags_pred = model(words)
         loss = F.binary_cross_entropy(tags)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         predictions += list(tags_pred.detach().numpy())
    print("Epoch: %d, loss: %.5f" % (epoch+1, loss.item()))


def test(model, test):
    test_losses = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in zip(test["word"], test["tag"]):
            for (word, tag) in zip(data, target):
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))

word_to_ix = {}

def word_index(data):
    for sentence in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def encoding_labels(data):
    possible = set(data)
    possible_labels = list(possible)
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    return label_dict

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def encode_dataset(data):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoded_data = tokenizer(data,
    padding=True,
    return_tensors='pt',)

    input_ids = encoded_data["input_ids"]

    model = BertModel.from_pretrained('bert-base-cased')
    return model

def lstm_train(word_to_ix, labeldict, trainset, embedder):
    model_lstm = LSTM(6, len(word_to_ix), len(labeldict), embedder)
    loss_function = nn.NLLLoss()
    optimizer_lstm = optim.SGD(model_lstm.parameters(), lr=0.1)

    for epoch in range(5):
        for (words, tags) in zip(trainset["word"], trainset["tag"]):

                model_lstm.zero_grad()
                sentence_in = prepare_sequence(words, word_to_ix)
                targets = prepare_sequence(tags, labeldict)
                tag_scores = model_lstm(sentence_in)

                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer_lstm.step()
    with torch.no_grad():
        inputs = prepare_sequence(trainset["word"], word_to_ix)
        tag_scores = model_lstm(inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Task 2")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")

    args = parser.parse_args()
    dataset = load_dataset("loading_script.py")
    trainset = dataset["train"]
    testset = dataset["test"]
    trainloader = torch.utils.data.DataLoader(trainset,
                                          shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,
                                         shuffle=False, num_workers=0)

    word_index = word_to_ix(trainset["word"])
    label = encoding_labels(trainset["tag"])
    emb = encode_dataset(trainset)
    tagsize = len(word_index.keys())
    vocabsize = len(label.keys())

    train_cnn(TextClassifier(), trainset, word_to_ix, label)
    test(TextClassifier(), testset)
    lstm_train(word_index, label, trainset, emb)
    test(LSTM(6, vocabsize, tagsize, emb), test)
