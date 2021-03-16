import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_bert_embed_matrix(train, test):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoded_data_train = tokenizer(train,
    padding=True,
    return_tensors='pt',)
    encoded_data_test = tokenizer(test,
    padding=True,
    return_tensors='pt',)

    input_ids_train = encoded_data_train["input_ids"]
    input_ids_test = encoded_data_test["input_ids"]
    bert = BertModel.from_pretrained('bert-base-cased')
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, embedder):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = embedder(embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words):
        embeds = self.word_embeddings(words)
        lstm_out, _ = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
