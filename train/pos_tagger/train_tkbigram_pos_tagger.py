from transformers import TFBertForTokenClassification
from tensorflow.keras.layers import LSTM,Input,Dropout, Bidirectional,Embedding,TimeDistributed,Dense
from tensorflow.keras.models import Model
from transformers import create_optimizer
from transformers import TFBertModel

import pickle
import tensorflow_addons as tfa
import numpy as np
import datetime
import tensorflow as tf
import pickle
import os
import argparse

with open('kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
tag_len = len(tag_dict.keys())

parser = argparse.ArgumentParser(description="Postagger")

parser.add_argument("--MAX_LEN",type=int,help="MAX Sequnce Length",default=200)
parser.add_argument("--BATCH",type=int,help="BATCH Size",default=50)
parser.add_argument("--EPOCH",type=int,help="EPOCH Size",default=5)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=240)
parser.add_argument("--hidden_state",type=int,help="BiLstm Hidden State",default=tag_len*2)
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="tkbigram_one_first_alltag_bert_tagger.model")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM

EPOCH = args.EPOCH
count_data = args.epoch_step#4000
validation_step = args.validation_step

max_len = args.MAX_LEN
def dataset():
    for _ in range(EPOCH):
        for i in range(count_data):
            masks = []
            segments = []
            
            data = np.load('kcc150_data/%05d_x.npy' % i)
            y = np.load('kcc150_data/%05d_y.npy' % i)
            bi = np.load('kcc150_data/%05d_tk_bigram.npy' % i)
            
            for d in data:
                d_ = [t for t in d if t != 1]
                mask = [1]*len(d_) + [0] * (max_len-len(d_))
                masks.append(mask)
                segment = [0] * max_len
                segments.append(segment)
            masks = np.array(masks)
            segments = np.array(segments)
            yield [bi,data,masks,segments],y

def validation():
    for _ in range(EPOCH):
        for i in range(count_data,count_data+240):
            masks = []
            segments = []
            data = np.load('kcc150_data/%05d_x.npy' % i)
            y = np.load('kcc150_data/%05d_y.npy' % i)
            bi = np.load('kcc150_data/%05d_tk_bigram.npy' % i)
            for d in data:
                d_ = [t for t in d if t != 1]
                mask = [1]*len(d_) + [0] * (max_len-len(d_))
                masks.append(mask)
                segment = [0] * max_len
                segments.append(segment)
            masks = np.array(masks)
            segments = np.array(segments)
            yield [bi,data,masks,segments],y


with open('kcc150_all_tokenbi.pkl','rb') as f:
    bigram = pickle.load(f)

class PosTaggerModel:
    def __init__(self,max_len,hidden_state,tag_len):
        self.max_len = max_len
        self.hidden_state = hidden_state
        self.tag_len = tag_len

    def build(self):
        bmodel = TFBertModel.from_pretrained('monologg/kobert', from_pt=True)#TFBertForTokenClassification.from_pretrained("monologg/kobert", num_labels=32, from_pt=True)

        input_bigram = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_bigram_ids')
        emb = Embedding(len(bigram.keys()),80,input_length=self.max_len)(input_bigram)
        emb_lstm = LSTM(768)(emb)
        emb_lstm = tf.expand_dims(emb_lstm,1)

        token_inputs = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_word_ids')
        mask_inputs = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_masks')
        segment_inputs = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_segment')
        outputs = bmodel.bert(input_ids=token_inputs,attention_mask = mask_inputs,token_type_ids = segment_inputs)[0]

        outputs = tf.keras.layers.Concatenate(axis=-2)([emb_lstm, outputs[:,1:,:]])

        lstm = Bidirectional(LSTM(self.hidden_state,return_sequences=True,dropout=0.1))(outputs)
        lstm = TimeDistributed(Dense(self.tag_len, activation='softmax'))(lstm)


        model = Model(inputs=[input_bigram,token_inputs,mask_inputs,segment_inputs],outputs=lstm)
        # model.summary()
        # Rectified Adam 옵티마이저 사용

        # 총 batch size * 4 epoch = 2344 * 4
        opt = tfa.optimizers.RectifiedAdam(lr=2.0e-5, total_steps = 2344*4
                , warmup_proportion=0.1, min_lr=1e-7, epsilon=1e-08, clipnorm=1.0)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-7)
        opt, schedule = create_optimizer(
                    init_lr=8e-5, 
                    num_warmup_steps=0, 
                    num_train_steps=count_data
                )

        
        model.compile(optimizer=opt,     
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=tf.metrics.SparseCategoricalAccuracy())

        return model

n_data = dataset()
n_validation = validation()

log_dir = "logs/" + datetime.datetime.now().strftime("pretag_%Y%m%d-%H%M%S")
model = PosTaggerModel(max_len,args.hidden_state,tag_len).build()
model.fit(n_data,epochs=EPOCH,batch_size=args.BATCH,steps_per_epoch=count_data,validation_data=n_validation,validation_steps=validation_step,callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])

model.save(args.model_name)
