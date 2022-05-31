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

model = tf.keras.models.load_model('lstm_bigram_tokenizer.model',compile=False)
model2 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_tagone.model',compile=False)
# print('model 1')
# model.summary()

# print('model 2')
# model2.summary()
import time

time.sleep(120)
