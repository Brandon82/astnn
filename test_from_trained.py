import torch
import pandas as pd
from model import BatchProgramClassifier
from gensim.models.word2vec import Word2Vec
import numpy as np

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2]-1)
#        p_set.append(item[5])
    return data, torch.LongTensor(labels)


word2vec = Word2Vec.load('data/embedding/train/node_w2v_128').wv
embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

# Load the test data
test_data = pd.read_pickle('./data/split_data/test/blocks.pkl')

HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 2
EPOCHS = 16
BATCH_SIZE = 4
USE_GPU = False
MAX_TOKENS = word2vec.vectors.shape[0]
EMBEDDING_DIM = word2vec.vectors.shape[1]

# Load the model
model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                USE_GPU, embeddings)

model.load_state_dict(torch.load('./data/saved_model/model.pt'))
model.eval()



i = 0
torch.cuda.empty_cache()
batch = get_batch(test_data, i, BATCH_SIZE)
i += BATCH_SIZE
train_inputs, train_labels = batch
if USE_GPU:
    train_inputs, train_labels = train_inputs, train_labels.cuda()

model.zero_grad()
model.batch_size = len(train_labels)
model.hidden = model.init_hidden()
output = model(train_inputs)

# calc training acc
_, predicted = torch.max(output.data, 1)

# Print the predicted label
print(predicted)

