import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2]-1)
#        p_set.append(item[5])
    return data, torch.LongTensor(labels)

if __name__ == '__main__':
    
    root = 'data/'
    train_data = pd.read_pickle(root+'train/blocks.pkl')
    val_data = pd.read_pickle(root + 'dev/blocks.pkl')
    test_data = pd.read_pickle(root+'test/blocks.pkl')

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    EPOCHS = 15
    BATCH_SIZE = 32
    USE_GPU = False
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    
    output = 'Final-' + str(EPOCHS)
    if not os.path.isdir(os.getcwd() + '/Results/' + output):
        os.makedirs(os.getcwd() + '/Results/' + output)
    data_table = []
    data_out = 'Results/' + output + '/' + output + '.csv'
    stats_out = 'Results/' + output + '/' + output + '-stats.xlsx'

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    
    train_recall_ = []
    val_recall_ = []
    train_precision_ = []
    val_precision_ = []
    train_f1_ = []
    val_f1_ = []
    
    stats_ = []
    
    tpos_ = []
    tneg_ = []
    fpos_ = []
    fneg_ = []
    
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        
        tpos = [0]*2
        tneg = [0]*2
        fpos = [0]*2
        fneg = [0]*2
        
        i = 0
        while i < len(train_data):
            torch.cuda.empty_cache()
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

            for p, t, s in zip(predicted, train_labels):
                if p == t == 1:
                    tpos[0] += 1
                elif p == t == 0:
                    tneg[0] += 1
                elif p != t == 0:
                    fpos[0] += 1
                elif p != t == 1:
                    fneg[0] += 1

        
        
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        
        try:
            train_recall_.append(tpos[0] / (tpos[0] + fneg[0]))
        except ZeroDivisionError:
            train_recall_.append(0)
        try:
            train_precision_.append(tpos[0] / (tpos[0] + fpos[0]))
        except ZeroDivisionError:
            train_precision_.append(0)
        try:
            train_f1_.append(2 * train_recall_[epoch] * train_precision_[epoch] / (train_recall_[epoch] + train_precision_[epoch]))
        except ZeroDivisionError:
            train_f1_.append(0)

        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        
        i = 0
        while i < len(val_data):
            torch.cuda.empty_cache()
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels, val_sets = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
            
            for p, t, s in zip(predicted, val_labels, val_sets):
                if p == t == 1:
                    tpos[1] += 1
                elif p == t == 0:
                    tneg[1] += 1
                elif p != t == 0:
                    fpos[1] += 1
                elif p != t == 1:
                    fneg[1] += 1
            
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        
        try:
            val_recall_.append(tpos[1] / (tpos[1] + fneg[1]))
        except ZeroDivisionError:
            val_recall_.append(0)
        try:
            val_precision_.append(tpos[1] / (tpos[1] + fpos[1]))
        except ZeroDivisionError:
            val_precision_.append(0)
        try:
            val_f1_.append(2 * val_recall_[epoch] * val_precision_[epoch] / (val_recall_[epoch] + val_precision_[epoch]))
        except ZeroDivisionError:
            val_f1_.append(0)
            
        tpos_.append(tpos[:])
        tneg_.append(tneg[:])
        fpos_.append(fpos[:])
        fneg_.append(fneg[:])
        
        stats_.append([tpos[:], tneg[:], fpos[:], fneg[:]])
            
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))
        
        # save data
        tmp_str = ('%3d %.4f %.4f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'
              % (epoch + 1, train_loss_[epoch], val_loss_[epoch], train_acc_[epoch], val_acc_[epoch], (end_time - start_time), train_recall_[epoch], train_precision_[epoch], train_f1_[epoch], val_recall_[epoch], val_precision_[epoch], val_f1_[epoch]))
        tmp_table = tmp_str.split()
        data_table.append(tmp_table)

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        torch.cuda.empty_cache()
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels, test_sets = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
        
    df = pd.DataFrame(data_table, columns = 
                      ['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy', 'Time Cost (s)', 'Training Recall', 'Training Precision', 'Training F1 Score', 'Validation Recall', 'Validation Precision', 'Validation F1 Score'])
    
    writer = pd.ExcelWriter(stats_out, engine = 'xlsxwriter')
    for i in range(EPOCHS):
        tmp = pd.DataFrame(stats_[i], columns = ['Training', 'Validation'], index=None)
        tmp.insert(0, 'Stat', ['True Positive', 'True Negative', 'False Positive', 'False Negative'], True)
        tmp.to_excel(writer, sheet_name = str(i), index=None)
    writer.save()
        
    df.to_csv(path_or_buf=data_out, index=None)
    print('Done.')