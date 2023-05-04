import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root = __file__
root = root.split(os.sep)
root = os.sep.join(root[:-1])
os.environ['HT'] = root
tok_max_len = 300
# print('ht',root)
# from urllib import request
# from flask import Flask,request, render_template
import tensorflow as tf
from keras import backend as K
# from keras.models import load_model
# import threading as t
import numpy as np
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
import threading, requests, time
import threading
from multiprocessing import Pool, Process
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
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
tok_model = None
pos_model = None
# mod = None
tok_model2 = None
tok_model3 = None
pos_model2 = None
pos_model3 = None
def thread_tok():
    global tok_model, tok_model2, tok_model3
    tok_model = tf.keras.models.load_model(root + os.sep + 'lstm_bigram_tokenizer.model',compile=False)
    # tok_model2 = tf.keras.models.load_model(root + os.sep + 'lstm_bigram_tokenizer.model',compile=False)
    # tok_model3 = tf.keras.models.load_model(root + os.sep + 'lstm_bigram_tokenizer.model',compile=False)
# from tensorflow import Graph, Session
# thread_graph = Graph()
# with thread_graph.as_default():
#     thread_session = Session()
#     with thread_session.as_default():
def thread_pos():
    global pos_model,pos_model2,pos_model3
    pos_model = tf.keras.models.load_model(root + os.sep + 'tkbigram_one_first_alltag_bert_tagger_distil.model',compile=False)
    # pos_model2 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_electra.model',compile=False)
    # pos_model3 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_electra.model',compile=False)
    # pos_model3 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_electra.model',compile=False)
    # pos_model = tf.keras.models.load_model(root + os.sep + 'tkbigram_one_first_alltag_bert_tagger_distil.model',compile=False)
        # graph = tf.get_default_graph()
        # sess = tf.get_default_session()



thread_tok()
thread_pos()
# self.cnn_model = load_model(model_path)
# self.cnn_model.predict(np.array([[0,0]])) # warmup
# K.clear_session()
# session = K.get_session()
# graph = tf.get_default_graph()
# graph.finalize() # finalize
end_t = time.time()
print("모델 로딩 {}".format((end_t-start_t)))
# end = time.time()
# print(end-start)
def get_tok(line,result,tok_model,verbose=0):
    # line = line.replace(' ','▁')
    # if model == None:
    # with session.as_default():
    #     with graph.as_default():
    tok = tokenizer.predict(tok_model,line,verbose=verbose)
    # token = tok[0].split(' ')
    # result = []
    result['1'] = tok
    # for t in token:
    #     t = t.split('+')
    #     result.append(t)

    # return tok

def get_pos(tok,tok_line,result,pos_model,verbose=0):
    # with session.as_default():
    #     with graph.as_default():
    pos = tagging.predictbi(pos_model,tok,tok_line,lite=True,verbose=verbose)
    result['pos'] = pos
    # x1 = tok[:100]
    # x2 = tok[100:]
    # tkline1 = tok_line[:100]6
    # tkline2 = tok_line[100:]
    # pos = tagging.start(x1,x2,tkline1,tkline2)
    # postag = pos[0].split(' ')
    # result = []
    # for p in postag:
    #     p = p.split('+')
    #     result.append(p)
    
    # return pos#result,postag
def get_tok2(line,result,verbose=0):
    # line = line.replace(' ','▁')
    # if model == None:
    # with session.as_default():
    #     with graph.as_default():
    tok = tokenizer.predict(tok_model,line,verbose=verbose)
    result['2'] = tok
    # token = tok[0].split(' ')
    # result = []
    # for t in token:
    #     t = t.split('+')
    #     result.append(t)

    # return tok
def get_pos2(tok,tok_line,result,verbose=0):
    # with session.as_default():
    #     with graph.as_default():    
    pos = tagging.predictbi(pos_model,tok,tok_line,lite=False,verbose=verbose)
    result['pos2'] = pos
    # x1 = tok[:100]
    # x2 = tok[100:]
    # tkline1 = tok_line[:100]6
    # tkline2 = tok_line[100:]
    # pos = tagging.start(x1,x2,tkline1,tkline2)
    # postag = pos[0].split(' ')
    # result = []
    # for p in postag:
    #     p = p.split('+')
    #     result.append(p)
    
    # return pos#result,postag
# def get_pos(tok,tok_line,verbose=0):
    
#     pos = tagging.predictbi(pos_model,tok,tok_line,lite=False,verbose=verbose)
thread_num = 1
batch = 300
models = [[tok_model,tok_model2,tok_model3],[pos_model,pos_model2,pos_model3]]
from tqdm import tqdm
def preprocess(l):
    temp_l = []
    l_temp = 'ㄱ'
    dot_flag = False
    # l = re.sub('[0-9]+\.[0-9]+\.[0-9]+','datetime',l)
    # l = re.sub('[0-9]+\.[0-9]+','float',l)
    # l = re.sub('[0-9]+','number',l)
    for index,ll in enumerate(l):
        ll = ll.lower()
        if '가' <= ll <= '힣' \
            or 'ㄱ' <= ll <= 'ㅎ' \
            or 'a' <= ll <= 'z' \
            or ll == '.' \
            or ll == ' ':
                if ll == '.' and not ('0' <= l_temp <='9'):
                    temp_l.append(' ')
                temp_l.append(ll)
        elif '0' <= ll <= '9':
            temp_l.append(ll)
            if len(l)-1 != index:
                if not ('0' <= l[index+1] <= '9'):
                    temp_l.append(' ')
        l_temp = ll
                #x_te[index_temp] = ' '#' '+x_te[index_temp] + ' '
    if len(l) > 0  and l[-1] != '.':
        temp_l.append(' .')
        dot_flag = True
    l = ''.join(temp_l)
    l = re.sub(' +',' ',l)
    # print('pre',l)
    return l, dot_flag

def analysis(x,verbose=0):
    if type(x) == str:
        xxx = [x]
    else:
        xxx = x    
    X = []
    for xx in xxx:
        x_,dot_flag = preprocess(xx)
        if dot_flag and len(xx) >= tok_max_len:
            xx = xx[:tok_max_len-2]
            xx,_ = preprocess(xx)
        else:
            xx = x_
                
        if tok_max_len > len(xx):
            xx = xx[:tok_max_len]
        
        X.append(xx.replace(' ','▁'))
    result = [{}]
    get_tok(X,result[0],tok_model,verbose=verbose)
    tok_ = []
    tok = result[0]['1']
    for tk in tok:
        tok_.append(tk.replace('+',' '))
    get_pos(tok_,tok,result[0],pos_model,verbose=verbose)
    # X = []
    pos = result[0]['pos']
    return pos

if __name__ == '__main__':

    while True:
        mode = input('mode file!!"filename",or text!!"text" or exit:')
        dot_flag = []
        args = mode.split('!!')
        mode = args[0]
        if len(args) > 1:
            filename = args[1]
            
        if mode == 'file':
            # filename = input('file name:')
            result_f = input('output file name:')
            inputf = open(filename,encoding='utf-8')
            output = open(result_f,'w',encoding='utf-8')
            output_tk = open('tk_'+result_f,'w',encoding='utf-8')
            count = 0
            for i in inputf:
                count += 1
            inputf.seek(0)
            X = []
            # thread_tok()
            t = 0
            with tqdm(total=count) as pbar:
                for _ in range(count):#input:
                    l = inputf.readline()
                    l = l.strip()
                    if tok_max_len > len(l):
                        l = l[:tok_max_len]
                    # if l[-1] != '.':
                    #     dot_flag.append(True)
                    # else:
                    #     dot_flag.append(False)
                    l_,df = preprocess(l)
                    dot_flag.append(df)
                    if dot_flag[-1] and len(l) >= tok_max_len:
                        l = l[:tok_max_len-2]
                        l,_ = preprocess(l)
                    else:
                        l = l_
                    
                    X.append(l.replace(' ','▁'))
                    # X.append(l)
                    if len(X) == batch:#2000:
                        # result = {}
                        # result2 = {}
                        
                        # result3 = {}
                        # result['1'] = []
                        # result['2'] = []
                        # tok = get_tok(X[:500])
                        # tok2 = get_tok2(X[500:])
                        # target=get_tok(X, result)
                        # t = threading.Thread(target=get_tok, args=(X[:300], result))
                        # t2 = threading.Thread(target=get_tok2, args=(X[300:], result))
                        # t.start()
                        # t2.start()

                        # while t.is_alive() or t2.is_alive():
                        #     1 == 1
                        # tok = get_tok(X,result)
                        tok_ = [[] for _ in range(thread_num)]
                        result = [{} for _ in range(thread_num)]
                        # threads 
                        start = 0
                        end = batch // thread_num
                        for model_num in range(thread_num):
                            # tok_ = []
                            # tok_2 = []
                            th = threading.Thread(target=get_tok, args=(X[start:end],result[model_num],models[0][model_num]))
                            start = end
                            end = end + (batch // thread_num)
                            # p2 = threading.Thread(target=get_tok, args=(X[500:],result2,tok_model2))
                            
                            th.start(); th.join()
                            # p2.start(); p2.join()
                        # pbar.update(batch)
                        # X = []
                        # continue
                        for model_num in range(thread_num):
                            # for tk in result2['1']:#+result['2']:#tok:
                            #     tok_2.append(tk.replace('+',' '))
                            # tok,tok_line,result,pos_model
                            for tk in result[model_num]['1']:#+result['2']:#tok:
                                tok_[model_num].append(tk.replace('+',' '))
                            th = threading.Thread(target=get_pos, args=(tok_[model_num],result[model_num]['1'],result[model_num],models[1][model_num]))
                            # p2 = threading.Thread(target=get_pos, args=(tok_2,result2['1'],result2,pos_model2))
                        # p3 = threading.Thread(target=get_pos, args=(tok_[400:600],result['1'][400:600],result3,pos_model3))

                            th.start(); th.join()
                            # p2.start(); p2.join()
                        # p3.start(); p3.join()

                        # while p1.is
                        # get_pos(tok_,result['1'], result)
                        # t = threading.Thread(target=get_pos, args=(tok_[:300], result['1'],result))
                        # t2 = threading.Thread(target=get_pos2, args=(tok_[300:], result['2'],result))
                        # t.start()
                        # t2.start()
                        # while t.is_alive() or t2.is_alive():
                        #     1 == 1
                        # # pos = get_pos(tok_,tok)
                        # t+=1
                        # pos = tok
                        X = []

                        
                        # pos_ = []
                        # for p, df in zip(pos,dot_flag):
                        #     if df:
                        #         p = ' '.join(p.split()[:-1])
                        #     pos_.append(p)
                        # pos = pos_
                        # pbar.update(500)
                        for model_num in range(thread_num):
                            output.write('\n'.join(result[model_num]['pos'])+'\n')
                        # output.write('\n'.join(result2['pos'])+'\n')
                        # break
                        pbar.update(batch)
                        # pbar.update(1000)
                if len(X) > 0:
                    # print('T',X)
                    result = {}
                    # result['1'] = []
                    # result['2'] = []
                    # tok = get_tok(X[:500])
                    # tok2 = get_tok2(X[500:])
                    target=get_tok(X, result)
                    tok_ = []
                    for tk in result['1']:
                        # print(tk)
                        tok_.append(tk.replace('+',' '))
                    pos = get_pos(tok_,result['1'])
                    # pos = tok
                    # print(pos)
                    # pos_ = []
                    # for p, df in zip(pos,dot_flag):
                    #     if df:
                    #         p = ' '.join(p.split()[:-1])
                    #     pos_.append(p)
                    # pos = pos_
                    output.write('\n'.join(result['pos'])+'\n')
                output.close()
            # with open('tk_'+result_f,encoding='utf-8') as f:
            #     # index = 0
            #     # end = 2000
            #     tok_ = []
            #     tok = []
            #     # thread_pos()
            #     with tqdm(total = count) as pbar:
            #         for l in f:
            #             l = l.strip()
            #             tok_.append(l.replace('+',' '))
            #             tok.append(l)
            #             if len(tok_) % 100 == 0:
            #                 pos = get_pos(tok_,tok)
            #                 output.write('\n'.join(pos)+'\n')
            #                 tok_ = []
            #                 tok = []
            #                 pbar.update(100)
            #                 # break
            #         if len(tok_) > 0:
            #             pos = get_pos(tok_,tok)
            #             output.write('\n'.join(pos)+'\n')
            #             tok_ = []
            #             tok = []
            #         output.close()
        elif mode == 'text':
            # line = input('text:')
            line = filename
            line = line.strip()
            line = line[:tok_max_len]
            pos = analysis(line)
            # # dot_flag = False
            # # if line[-1] != '.':
            # #     dot_flag = True
            # line_, dot_flag = preprocess(line)
            # if dot_flag and len(line) >= tok_max_len:
            #     line = line[:tok_max_len-2]
            #     line,_ = preprocess(line)
            # else:
            #     line = line_
            
            # # line,_ = preprocess(line)
            # line = [line.replace(' ','▁')]
            # tok = get_tok(line)
            # tok[0] = tok[0].replace('▁',' ')
            # tok_ = []
            # for tk in tok:
            #     tok_.append(tk.replace('+',' '))
            # pos = get_pos(tok_,tok)
            # # print(pos)
            # if dot_flag:
            #     pos = [' '.join(pos[0].split()[:-1])]
            print(pos[0])
        elif mode == 'exit':
            break
