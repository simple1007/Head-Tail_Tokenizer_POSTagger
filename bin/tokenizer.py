import numpy as np
import tensorflow as tf
import pickle
import re
import os
import math
root = os.environ['HT']
# root = '.'
max_len = 300
with open(root+'/lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open(root+'/bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

def w2i(x_t):
    res = []
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        x = list(x)
        x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 

        x = x

        x = x + [0] * (max_len-len(x))
        x = x[:max_len]
        res.append(x)

    x = np.array(res)

    return x

def predict(model,x,verbose=0):
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
        # if len(l) == '':
        #     bi_npy.append(bigram['_'])
        bi_npy = bi_npy + [0] * (max_len - len(bi_npy))
        bi_npy = np.array(bi_npy[:max_len])
        BI.append(bi_npy)
    x = w2i(x)
    x = np.array(x)
    
    BI = np.array(BI)
    # print(type(BI),BI.shape)
    result = None
    # try:
    result = model.predict([x,BI],verbose=verbose)
    # except Exception as ex:
    #     print(ex)
    #     # print(x_temp)
    #     with open('error.txt','w',encoding='utf-8') as f_:
    #         sh = []
    #         for xt,xx in zip(x_temp,BI):
    #             # if xx.shape[1] > 300:
    #             f_.write("{} {}".format(len(xt),xx.shape)+'\n')
    #             f_.write(xt+'\n')
    #             f_.write(str(xx)+'\n')
    #             sh.append(xx.shape[0])
    #         sh = tuple(sh)
    #     print(sh)
    #     print(ex)
    #     exit()
    result_ = tf.argmax(result,axis=2).numpy()
    tagging = []
    # print(result_.shape)
    # for i in result_:
    #     print(i)
    for index,tag_ in enumerate(result_):
        x_te = list(x_temp[index])
        # print(tag_)
        # print(x_te)
        tag_prob = []
        for index_temp,te in enumerate(tag_):
            # if index_temp <= len(x_te)-1:
            #     x_te[index_temp] = x_te[index_temp].lower()
            # if index_temp <= len(x_te)-1 \
            # and not '가' <= x_te[index_temp] <= '힣' \
            # and not 'ㄱ' <= x_te[index_temp] <= 'ㅎ' \
            # and not 'a' <= x_te[index_temp] <= 'z' :
            #     x_te[index_temp] = ' '#' '+x_te[index_temp] + ' '
            if te == 2 and index_temp <= len(x_te)-1:
                x_te[index_temp] = x_te[index_temp] + '+' + '&&' + str(result[index][index_temp][2]) + '&&'
                # tag_prob.append([index_temp,result[index][index_temp]])
            elif te == 3 and index_temp < len(tag_)-1 and tag_[index_temp+1] == 3:
                break
        x_te = ''.join(x_te).replace('▁',' ')
        # print(x_te)
        x_te = x_te.split(' ')
        temp_tok = []
        for ti,xttt in enumerate(x_te):
            if xttt.count('+') > 1:
                # print('fff',xttt)
                # print(tag_prob)   
                xttt = xttt.split('&&')
                mem = []
                mem_prob = []
                max = 0
                maxi = -1
                # print(xttt)
                for memi, xttt_ in enumerate(xttt):
                    if memi % 2 == 0:
                        mem.append(xttt_.strip('+'))
                        continue
                    else:
                        # print('fff',xttt_)
                        # continue
                        1 == 1
                        # mem.append(xttt_.split('&&')[0])
                    
                    xttt_ = xttt_.split('&&')
                    # print(xttt_)
                    # maxi = memi
                    
                    if max < float(xttt_[0]):
                        max = float(xttt_[0])
                        maxi = memi - len(mem) + 1
                        # print(max,maxi)
                        
                    # mem_prob(float(xttt_[1]))
                # print(mem,maxi)
                h = ''.join(mem[:maxi])#[0]
                # t = ''.join(xttt[1:])
                t = ''.join(mem[maxi:])
                xttt = h+'+'+t
            temp_tok.append(xttt)
            # else:
            #     temp_tok.append(xttt)
        x_te = ' '.join(temp_tok)
        x_te = re.sub(' +',' ',x_te)
        # print(x_te)
        x_te = re.sub('&&[0-9].[0-9]+&&','',x_te)
        # print(x_te)
        x_te = x_te.strip()
        tagging.append(x_te)
    # print(tagging)
    return tagging