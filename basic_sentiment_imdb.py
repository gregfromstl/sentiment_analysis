import torch
from torchtext import data
from torchtext import datasets
import random

SEED = 1234
MAX_VOCAB_SIZE = 25000

# make experiment reproducible
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

print(TEXT.vocab.freqs.most_common(20))
