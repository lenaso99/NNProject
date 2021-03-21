import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch
import math


class TextClassifier(nn.ModuleList):
    def __init__(self):
      super(TextClassifier, self).__init__()
      self.embed = BertModel.from_pretrained('bert-base-cased')

      self.dropout = nn.Dropout(0.25)
      self.kernel_1 = 2
      self.kernel_2 = 3
      self.kernel_3 = 4
      self.kernel_4 = 5

      self.embedding = self.embed(padding_idx=0)

      self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
      self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
      self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
      self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

      self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
      self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
      self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
      self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

      self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):

          out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
          out_conv_1 = math.floor(out_conv_1)
          out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
          out_pool_1 = math.floor(out_pool_1)

          out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
          out_conv_2 = math.floor(out_conv_2)
          out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
          out_pool_2 = math.floor(out_pool_2)

          out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
          out_conv_3 = math.floor(out_conv_3)
          out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
          out_pool_3 = math.floor(out_pool_3)

          out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
          out_conv_4 = math.floor(out_conv_4)
          out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
          out_pool_4 = math.floor(out_pool_4)

          return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

    def forward(self, x):
      x = self.embedding(x)

      x1 = self.conv_1(x)
      x1 = torch.relu(x1)
      x1 = self.pool_1(x1)

      x2 = self.conv_2(x)
      x2 = torch.relu((x2))
      x2 = self.pool_2(x2)

      x3 = self.conv_3(x)
      x3 = torch.relu(x3)
      x3 = self.pool_3(x3)

      x4 = self.conv_4(x)
      x4 = torch.relu(x4)
      x4 = self.pool_4(x4)

      union = torch.cat((x1, x2, x3, x4), 2)
      union = union.reshape(union.size(0), -1)

      out = self.fc(union)
      out = self.dropout(out)
      out = torch.sigmoid(out)

      return out.squeeze()


class LSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, tagset_size, embedder):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = embedder

        self.lstm = nn.LSTM(hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words):
        embeds = self.word_embeddings(**words)
        lstm_out, _ = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
