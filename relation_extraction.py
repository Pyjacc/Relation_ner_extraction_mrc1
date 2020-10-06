import numpy as np
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # '-1' means use cpu

num_classes = 44
maxlen = 350
batch_size = 16     #batch_size设置大后可能出现OOM问题
# 数据集下载网址：https://linux.ctolib.com/brightmart-albert_zh.html
config_path = './pre_train/albert/albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = './pre_train/albert/albert_small_zh_google/albert_model.ckpt'
dict_path = './pre_train/albert/albert_small_zh_google/vocab.txt'


def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            l_list = l.split('jjjj')
            text = l_list[0]
            labels = l_list[1]
            D.append((text,labels))
    return D



# 加载数据集
train_data = load_data('./datasets/train_data/re/relation_datasets_train.csv')
valid_data = load_data('./datasets/train_data/re/relation_datasets_valid.csv')


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True) #如果涉及关于数字，字母粒度得操作，最好重构token，比如NER模型


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            label = label.split(' ')        #多个标签间用空格拆分
            label_ = [0] * num_classes
            for k in label:
                label_[int(k)] = 1
            batch_labels.append(label_)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)


output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='sigmoid',       #多标签使用sigmod
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)

def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return micro_f1, macro_f1


def evaluate(data):
    pass



class Evaluator(keras.callbacks.Callback):
    pass


def predict_re(text):
    pass


if __name__ == "__main__":
    # train
    evaluator = Evaluator()
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=40,
        callbacks=[evaluator]
    )


    #test
    text = '产后抑郁症@### 睡眠剥夺 研究提示婴儿睡眠模式与产妇疲劳以及产后新发抑郁症状间存在强关联性。'
    res = predict_re(text)
    print(res)