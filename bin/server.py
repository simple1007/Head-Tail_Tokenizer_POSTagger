# from urllib import request
from flask import Flask,request, render_template
import tokenizer
import tagging
import json
import re
app = Flask(__name__)

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

tok_model = tf.keras.models.load_model('lstm_bigram_tokenizer.model',compile=False)
pos_model = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_tagone.model',compile=False)

def get_tok(line):
    line = line.replace(' ','▁')
    tok = tokenizer.predict(tok_model,[line])
    token = tok[0].split(' ')
    result = []
    for t in token:
        t = t.split('+')
        result.append(t)

    return result,tok

def get_pos(tok):
    pos = tagging.predictbi(pos_model,[tok[0].replace('+',' ')],tok)
    postag = pos[0].split(' ')
    result = []
    for p in postag:
        p = p.split('+')
        result.append(p)
    
    return result,' '.join(postag)

@app.route('/tokenizer/<analyze>',methods=["POST"])
def tok(analyze):
    line = request.get_json()['line']
    line = line.replace('.',' . ').replace(',',' , ').replace('!',' ! ').replace('?',' ? ').replace('"',' " ').replace('\'',' \' ').replace('&',' & ').replace('%',' % ').replace('(',' ( ').replace(')',' ) ').replace('[',' [ ').replace(']',' ] ')
    line = re.sub(' +',' ',line)
    line = line.strip()
    
    token,token_l = get_tok(line)
    result = {}
    result['token'] = token_l[0]
    if analyze == 'pos':
        pos,postag = get_pos(token_l)
        # result = {}
        result['pos'] = postag
        #pos = tagging.predictbi(pos_model,[tok[0].replace('+',' ')],tok)
    
    response = app.response_class(
        response=json.dumps(result,ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    
    return response

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
    # token,token_l = get_tok('나는 밥을 먹고 학교에 갔다')
    # pos,postag = get_pos(token_l)

    # print(token,token_l)
    # print(pos,postag)