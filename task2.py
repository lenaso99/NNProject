import argparse
import nn_classes
from nn_classes import CNN, LSTM, get_bert_embed_matrix
import torch.optim as optim
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset

#TODO: import all other files and write code to execute them


dataset = load_dataset("loading_script.py")
cnn_network = CNN()
optimizer = optim.SGD(cnn_network.parameters(), lr=0.01,
                      momentum=0.5, weight_decay=1e-5)

trainset = dataset["train"]
testset = dataset["test"]
trainloader = torch.utils.data.DataLoader(trainset,
                                      shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset,
                                     shuffle=False, num_workers=0)

train_losses = []
train_counter = []
test_losses = []
epochs = 3
test_counter = [i*len(trainloader.dataset) for i in range(epochs + 1)]


def cnn_train(epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data["word"]
        labels = list(set(data["tag"]))
        optimizer.zero_grad()

  # cnn_network.cnn_train()
  # for batch_idx, (data, target) in enumerate(trainloader):
  #   optimizer.zero_grad()
  #   output = cnn_network(data)
  #   loss = F.nll_loss(output, target)
  #   loss.backward()
  #   optimizer.step()
  #   if batch_idx % args.log_interval == 0:
  #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
  #       epoch, batch_idx * len(data), len(trainloader.dataset),
  #       100. * batch_idx / len(trainloader), loss.item()))
  #     train_losses.append(loss.item())
  #     train_counter.append(
  #       (batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))
  #     torch.save(cnn_network.state_dict(), 'C:\\Users\\Leoni\\results\\model.pth')
  #     torch.save(optimizer.state_dict(), 'C:\\Users\\Leoni\\results\\optimizer.pth')

def cnn_test():
  cnn_network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in testloader:
      output = cnn_network(data)
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
    for word in data:
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


def lstm_train(word_to_ix, labeldict, trainset):
    model_lstm = LSTM(6, 6, len(word_to_ix), len(labeldict), get_bert_embed_matrix(trainset["word"], testset["word"]))
    loss_function = nn.NLLLoss()
    optimizer_lstm = optim.SGD(model_lstm.parameters(), lr=0.1)

    for epoch in range(5):
        for words in trainset["word"]:
            for tags in trainset["tag"]:

                model_lstm.zero_grad()
                sentence_in = prepare_sequence(words, word_to_ix)
                targets = prepare_sequence(tags, labeldict)
                tag_scores = model_lstm(sentence_in)

                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()
    with torch.no_grad():
        inputs = prepare_sequence(trainset["word"], word_to_ix)
        tag_scores = model_lstm(inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Task 2")
    parser.add_argument("--tsv", dest="tsv", type=str, default="sequences.tsv")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    parser.add_argument("--learning_rate", dest="learning_rate", type=str, default="0.01")
    parser.add_argument("--momentum", dest="momentum", type=str, default="0.5")
    parser.add_argument("--epochs", dest="epochs", type=str, default="3")
    parser.add_argument("--log_interval", dest="log_interval", type=str, default="10")

    args = parser.parse_args()
    cnn_train(3)
    dataset = load_dataset("loading_script.py")
    trainset = dataset["train"]
    testset = dataset["test"]
    word_index = word_to_ix(trainset["word"])
    label = encoding_labels(trainset["tag"])
    lstm_train(word_index, label, trainset)
