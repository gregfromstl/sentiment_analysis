import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import pandas as pd
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor

torch.manual_seed(123)

train_file = 'data/train.tsv'
validation_percent = 0.1
test_file = ''
using_word_pairs = False

EMBEDDING_DIM = 10

class PandasDataset(Dataset):
  def __init__(self, dataframe):
    self.dataframe = dataframe

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    return self.dataframe.iloc[idx]['inputs'], self.dataframe.iloc[idx]['labels']

# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.labels, self.inputs = [], []
        self.test_labels, self.test_inputs = [], []

        self.prepare_train_data()
        if len(test_file) > 0:
          self.prepare_test_data()
        #https://stackoverflow.com/questions/57767854/keras-preprocessing-text-tokenizer-equivalent-in-pytorch
        self.train_data, self.val_data, self.test_data = self.load_datasets()
        all_words = []
        for document in self.train_data['inputs']:
            all_words.extend(document.split())
        vocab = set(all_words)
        self.vocab_size = len(vocab)

        self.embeddings = nn.Embedding(self.vocab_size, EMBEDDING_DIM)
        self.linear1 = nn.Linear(10000 * EMBEDDING_DIM, 128)
        self.linear2 = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def prepare_train_data(self):
        print("Loading training data...")
        with open(train_file, encoding='utf-8') as data:
          reader = csv.reader(data, delimiter='\t')
          idx = 0
          for row in reader:
            if len(row) == 2 and idx < 10:
                self.labels.append(int(row[0]))
                self.inputs.append(row[1])
                idx += 1
        print("Loaded {} documents".format(len(self.labels)))

    def prepare_test_data(self):
        print("Loading testing data...")
        with open(test_file, encoding='utf-8') as data:
          reader = csv.reader(data, delimiter='\t')
          for row in reader:
            if len(row) == 2:
                self.test_labels.append(int(row[0]))
                self.test_inputs.append(row[1])
        print("Loaded {} documents".format(len(self.test_labels)))

    def load_datasets(self):
        data = {'labels': self.labels, 'inputs': self.inputs}
        train_data = pd.DataFrame(data=data)
        train_data, val_data = train_test_split(train_data, test_size=validation_percent)
        test_data = {'labels': self.test_labels, 'inputs': self.test_inputs}
        test_data = pd.DataFrame(data=data)
        return train_data, val_data, test_data

    def generate_encodings(self, data, labels):
        encoder = StaticTokenizerEncoder(data, tokenize=lambda s: s.split(), min_occurrences=3)
        encoded_data = [encoder.encode(document) for document in data]
        encoded_data = [pad_tensor(x, length=10000) for x in encoded_data]
        data = {'labels': labels, 'inputs': encoded_data}
        return pd.DataFrame(data=data)

    def train_dataloader(self):
        encodings = self.generate_encodings(self.train_data['inputs'], self.train_data['labels'])
        return DataLoader(PandasDataset(encodings), batch_size=64)

    def val_dataloader(self):
        encodings = self.generate_encodings(self.val_data['inputs'], self.val_data['labels'])
        return DataLoader(PandasDataset(encodings), batch_size=64)

    def test_dataloader(self):
        encodings = self.generate_encodings(self.test_data['inputs'], self.test_data['labels'])
        return DataLoader(PandasDataset(encodings), batch_size=64)

    def forward(self, x):
        batch_size = x.shape[0]
        embeds = self.embeddings(x)
        out = F.relu(self.linear1(embeds.reshape(batch_size, 10000*EMBEDDING_DIM)))
        out = self.linear2(out)
        return self.activation(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.binary_cross_entropy(y_hat, y.type(torch.cuda.FloatTensor))
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.binary_cross_entropy(y_hat, y.type(torch.cuda.FloatTensor))
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

model = LitClassifier()
trainer = pl.Trainer(max_epochs=30, gpus=1)
trainer.fit(model)


