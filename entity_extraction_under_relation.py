import os
import numpy as np
from bert4keras.backend import keras,K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding,DataGenerator,ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import json
import tensorflow as tf
from keras.layers import Input,Lambda
from keras.utils import to_categorical
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


ELECTRA_CONFIG_PATH = "./pre_train/electra/chinese_electra_small_L-12_H-256_A-4/electra_config.json"
ELECTRA_CHECKPOINT_PATH = "./pre_train/electra/chinese_electra_small_L-12_H-256_A-4/electra_small"
ELECTRA_VOCAB_PATH = "./pre_train/electra/chinese_electra_small_L-12_H-256_A-4/vocab.txt"

#该类主要解决在识别一些英文和数字的时候，尽量按照一个个字符处理
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') #space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') #剩余字符是[UNK]
        return R

maxlen = 350
epochs = 40
batch_size = 32
bert_layer = 12
learning_rate = 1e-5
crf_lr_multiplier = 1000

model_path = './models/ner/best_model.weights'

# query_mapping = json.load(open('./mrc_query_mapping.json','r',encoding='utf-8'))

def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        lines = f.readlines()
    for l in lines:
        if not l:
            continue
        l = l.strip()
        l = l.split('jjjjj')
        query = l[0]
        text = l[1]
        start_idx = l[2].split(' ')
        end_idx = l[3].split(' ')
        D.append((query,text,start_idx,end_idx))
    return D

#标注数据
train_data = load_data('./datasets/train_data/ner/ner_datasets_train.csv')
valid_data = load_data('./datasets/train_data/ner/ner_datasets_valid.csv')


#建立分词器
tokenizer = OurTokenizer(ELECTRA_VOCAB_PATH,do_lower_case=True)

class data_generator(DataGenerator):
    def __iter__(self,random=False):
        batch_token_ids,batch_segment_ids,batch_start,batch_end = [],[],[],[]
        for is_end,(query,text,start_idx,end_idx) in self.sample(random):
            query_token_ids,query_segment_ids = tokenizer.encode(query)
            token_ids = query_token_ids.copy()
            start = query_segment_ids.copy()
            end = query_segment_ids.copy()
            w_token_ids = tokenizer.encode(text)[0][1:-1]
            text_len = len(w_token_ids)
            if len(token_ids) + len(w_token_ids) < maxlen:
                token_ids += w_token_ids
                start_tmp = [0] * text_len
                end_tmp = [0] * text_len
                try:
                    for s_idx in start_idx:
                        start_tmp[int(s_idx)] = 1
                    for e_idx in end_idx:
                        end_tmp[int(e_idx)] = 1
                except:
                    print('index error')
                start += start_tmp
                end += end_tmp
            else:
                continue
            token_ids += [tokenizer._token_end_id]
            segment_ids = query_segment_ids + [1] * (len(token_ids) - len(query_segment_ids))
            start += [0]
            end += [0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_start.append(to_categorical(start,2))
            batch_end.append(to_categorical(end,2))
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_start = sequence_padding(batch_start)
                batch_end = sequence_padding(batch_end)
                yield [batch_token_ids,batch_segment_ids,batch_start,batch_end],None
                batch_token_ids,batch_segment_ids,batch_start,batch_end = [],[],[],[]

bert_model = build_transformer_model(
    config_path=ELECTRA_CONFIG_PATH,
    checkpoint_path=ELECTRA_CHECKPOINT_PATH,
    model='electra'
)


mask = bert_model.input[1]
# print(bert_model.input)
start_labels = Input(shape=(None,2),name="start-labels")
end_labels = Input(shape=(None,2),name="end-labels")

output_layers = 'Transformer-%s-FeedForward-Norm' % (bert_layer -1)
x = bert_model.get_layer(output_layers).output

start_output = Dense(2,activation='sigmoid',name='start')(x)
end_output = Dense(2,activation='sigmoid',name='end')(x)

start_output = Lambda(lambda x:x ** 2)(start_output)
end_output = Lambda(lambda x:x ** 2)(end_output)


start_model = Model(bert_model.input,start_output)
end_model = Model(bert_model.input,end_output)

model = Model(bert_model.input + [start_labels,end_labels],[start_output,end_output])
model.summary()

start_loss = K.binary_crossentropy(start_labels,start_output)
start_loss = K.mean(start_loss,2)
start_loss = K.sum(start_loss * mask) / K.sum(mask)

end_loss = K.binary_crossentropy(end_labels,end_output)
end_loss = K.mean(end_loss,2)
end_loss = K.sum(end_loss * mask) / K.sum(mask)

loss = start_loss + end_loss
model.add_loss(loss)
model.compile(optimizer=Adam(learning_rate))



def extract(qtext):
    v = qtext.split('jjjjj')[0]
    text = qtext.split('jjjjj')[1]

    query_tokens,query_segment_ids = tokenizer.encode(v)
    token_ids = query_tokens.copy()
    token_ids_w = tokenizer.encode(text)[0][1:-1]
    token_ids += token_ids_w
    token_ids += [tokenizer._token_end_id]
    segment_ids = query_segment_ids + [1] * (len(token_ids) - len(query_tokens))

    start_out = start_model.predict([[token_ids],[segment_ids]])[0][len(query_segment_ids):-1]
    end_out = end_model.predict([[token_ids],[segment_ids]])[0][len(query_segment_ids):-1]

    start = np.argmax(start_out,axis=1)
    end = np.argmax(end_out,axis=1)

    res = [int(k) + int(v) if int(k) + int(v) < 2 else 1 for k,v in zip(start,end)]
    return res



class Evaluate(keras.callbacks.Callback):
    def __init__(self, val):
        super().__init__()
        self.best_val_f1 = 0
        self.val = val
    def evaluate(self):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for (query,text,start_idx,end_idx) in tqdm(self.val):
            start = [0] * len(text)
            end = [0] * len(text)
            for s in start_idx:
                start[int(s)] = 1
            for e in end_idx:
                end[int(e)] = 1
            q_text = query + 'jjjjj' + text
            R = extract(q_text)
            T = [int(k) + int(v) if int(k) + int(v) < 2 else 1 for k,v in zip(start,end)]
            assert len(R) == len(T)
            R_and_T = 0
            for i in range(len(T)):
                if T[i] == R[i]:
                    if T[i] == 1:
                        R_and_T += 1
            X += R_and_T
            Y += sum(R)
            Z += sum(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def on_epoch_end(self, epoch, logs=None):

        f1, precision, recall = self.evaluate()

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./models/ner/best_model.weights')
        print(
            'valid: f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.5f\n' % (f1,precision,recall,self.best_val_f1)
        )

def predict_ner(query,text):
    model.load_weights(model_path)
    res = []
    query_tokens,query_segment_ids = tokenizer.encode(query)
    token_ids = query_tokens.copy()
    token_ids_w = tokenizer.encode(text)[0][1:-1]
    token_ids += token_ids_w
    token_ids += [tokenizer._token_end_id]
    segment_ids = query_segment_ids + [1] * (len(token_ids) - len(query_tokens))

    start_out = start_model.predict([[token_ids],[segment_ids]])[0][len(query_segment_ids):-1]
    end_out = end_model.predict([[token_ids],[segment_ids]])[0][len(query_segment_ids):-1]

    start = np.argmax(start_out,axis=1)
    end = np.argmax(end_out,axis=1)
    # assert sum(start) == sum(end)
    start = [i for i,j in enumerate(start) if j == 1]
    end = [i for i,j in enumerate(end) if j == 1]
    for s,e in zip(start,end):
        s = int(s)
        e = int(e)
        res.append(text[s:e])
    return res





if __name__ == "__main__":
    # evaluator = Evaluate(valid_data)
    # train_generator = data_generator(train_data,batch_size)
    # print(len(train_generator))
    # # for d in train_generator:
    # #     print(list(d))
    # #     print(len(d[0]))
    # #     print(len(d))
    # #     break
    #
    # model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )
    query = "根据药物治疗关系，找出药物"
    res = "。jjjjj0 6jjjjj2 10"
    text = "帕金森病@抗胆碱药（例如苯海索）和金刚烷胺在治疗轻度症状（尤其是震颤）时也可能有效，但是抗胆碱能药物的不良反应常限制了其在老年患者中的使用。"
    ner = predict_ner(query,text)
    print(ner)
