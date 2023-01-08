import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from urllib import request
# from flask import Flask,request, render_template
import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
import tokenizer
import tagging
import json
import re
import time
import numpy as np
# app = Flask(__name__)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 
if gpus: 
        try: # Currently, memory growth needs to be the same across GPUs 
            for gpu in gpus: 
                tf.config.experimental.set_memory_growth(gpu, True) 
                logical_gpus = tf.config.experimental.list_logical_devices('GPU') 
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs") 
        except RuntimeError as e: 
                # Memory growth must be set before GPUs have been initialized 
                print(e)


# graph = tf.get_default_graph()
# import tensorflow.keras.backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess  = tf.Session(config=config)
# graph = tf.get_default_graph()
# K.set_session(sess) 
# # import time
# # start = time.time()
# with graph.as_default():
print('Model 로딩 중입니다. 모델 로딩까지 2분이상 소요 될 수 있습니다.')
start_t = time.time()
tok_model = tf.keras.models.load_model('lstm_bigram_tokenizer.model',compile=False)
# from tensorflow import Graph, Session
# thread_graph = Graph()
# with thread_graph.as_default():
#     thread_session = Session()
#     with thread_session.as_default():
pos_model = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_distil.model',compile=False)
        # graph = tf.get_default_graph()
        # sess = tf.get_default_session()
end_t = time.time()
print("모델 로딩 {}".format((end_t-start_t)))
# end = time.time()
# print(end-start)
def get_tok(line):
    # line = line.replace(' ','▁')
    tok = tokenizer.predict(tok_model,line)
    token = tok[0].split(' ')
    result = []
    # for t in token:
    #     t = t.split('+')
    #     result.append(t)

    return tok

def get_pos(tok,tok_line):
    pos = tagging.predictbi(pos_model,tok,tok_line,lite=True)
    # x1 = tok[:100]
    # x2 = tok[100:]
    # tkline1 = tok_line[:100]
    # tkline2 = tok_line[100:]
    # pos = tagging.start(x1,x2,tkline1,tkline2)
    # postag = pos[0].split(' ')
    # result = []
    # for p in postag:
    #     p = p.split('+')
    #     result.append(p)
    
    return pos#result,postag
from tqdm import tqdm
def preprocess(l):
    temp_l = []
    for ll in l:
        ll = ll.lower()
        if '가' <= ll <= '힣' \
            or 'ㄱ' <= ll <= 'ㅎ' \
            or 'a' <= ll <= 'z' \
            or ll == '.' \
            or ll == ' ':
                if ll == '.':
                    temp_l.append(' ')
                temp_l.append(ll)
                #x_te[index_temp] = ' '#' '+x_te[index_temp] + ' '
    l = ''.join(temp_l)
    return l

while True:
    mode = input('mode file!!"filename",or text!!"text" or exit:')

    args = mode.split('!!')
    mode = args[0]
    if len(args) > 1:
        filename = args[1]
        
    if mode == 'file':
        # filename = input('file name:')
        result_f = input('output file name:')
        inputf = open(filename,encoding='utf-8')
        output = open(result_f,'w',encoding='utf-8')
        count = 0
        for i in inputf:
            count += 1
        inputf.seek(0)
        X = []
        for _ in enumerate(tqdm(range(count))):#input:
            l = inputf.readline()
            l = l.strip()
            if 300 < len(l):
                l = l[:300]

            l = preprocess(l)
            X.append(l.replace(' ','▁'))
            # X.append(l)
            if len(X) == 200:
                tok = get_tok(X)
                tok_ = []
                for tk in tok:
                    tok_.append(tk.replace('+',' '))
                pos = get_pos(tok_,tok)
                X = []
                output.write('\n'.join(pos)+'\n')
        if len(X) > 0:
            # print('T',X)
            tok = get_tok(X)
            tok_ = []
            for tk in tok:
                # print(tk)
                tok_.append(tk.replace('+',' '))
            pos = get_pos(tok_,tok)
            # print(pos)
            output.write('\n'.join(pos)+'\n')
        output.close()
    elif mode == 'text':
        # line = input('text:')
        line = filename
        line = line.strip()
        line = line[:300]
        line = preprocess(line)
        line = [line.replace(' ','▁')]
        tok = get_tok(line)
        tok[0] = tok[0].replace('▁',' ')
        tok_ = []
        for tk in tok:
            tok_.append(tk.replace('+',' '))
        pos = get_pos(tok_,tok)
        print(''.join(pos))
    elif mode == 'exit':
        break
