import numpy as np
import tensorflow as tf
import pickle
import re
max_len = 300
with open('lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open('bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

def w2i(x_t):
    res = []
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        x = list(x)
        x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 

        x = x

        x = x + [0] * (max_len-len(x))
        res.append(x)

    x = np.array(res)

    return x

def predict(model,x):
    BI = []
    x_temp = x
    x_len = len(x)
    for l in x:
        bi_npy = []
        for i in range(len(l)-1):
            bi = l[i:i+2]
            if bi in bigram:
                bi_npy.append(bigram[bi])
            else:
                bi_npy.append(bigram['[UNK]'])
        bi_npy = bi_npy + [0] * (max_len - len(bi_npy))
        bi_npy = np.array(bi_npy)
        BI.append(bi_npy)
    x = w2i(x)
    x = np.array(x)

    BI = np.array(BI)
    result = model.predict([x,BI])
    result_ = tf.argmax(result,axis=2).numpy()
    tagging = []
    # print(result_.shape)
    # for i in result_:
    #     print(i)
    for index,tag_ in enumerate(result_):
        x_te = list(x_temp[index])
        # print(x_te)
        for index_temp,te in enumerate(tag_):
            # if index_temp <= len(x_te)-1:
            #     x_te[index_temp] = x_te[index_temp].lower()
            # if index_temp <= len(x_te)-1 \
            # and not '가' <= x_te[index_temp] <= '힣' \
            # and not 'ㄱ' <= x_te[index_temp] <= 'ㅎ' \
            # and not 'a' <= x_te[index_temp] <= 'z' :
            #     x_te[index_temp] = ' '#' '+x_te[index_temp] + ' '
            if te == 2 and index_temp <= len(x_te)-1:
                x_te[index_temp] = x_te[index_temp] + '+'
            elif te == 3 and index_temp < len(tag_)-1 and tag_[index_temp+1] == 3:
                break
        x_te = ''.join(x_te).replace('▁',' ')
        x_te = x_te.split(' ')
        temp_tok = []
        for xttt in x_te:
            if xttt.count('+') > 1:
                xttt = xttt.split('+')
                h = xttt[0]
                t = ''.join(xttt[1:])
                xttt = h+'+'+t
            temp_tok.append(xttt)
        x_te = ' '.join(temp_tok)
        x_te = re.sub(' +',' ',x_te)
        x_te = x_te.strip()
        tagging.append(x_te)
    # print(tagging)
    return tagging