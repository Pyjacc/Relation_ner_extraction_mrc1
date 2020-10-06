'''
生成实体训练的数据集
（1）保存关系训练集样本和验证集样本
（2）保存ner训练样本（后续中使用）
'''

import random
import json
import os


def create_relation_ner_sample():
    f_w_relation = open('./datasets/train_data/relation_datasets.csv', 'w', encoding='utf-8')  # 保存关系训练样本
    f_w_ner = open('./datasets/train_data/ner_datasets.csv', 'w', encoding='utf-8')  # 保存ner训练样本
    query_mapping = json.load(open('./datasets/train_data/query_mapping.json', 'r', encoding='utf-8'))  # 导入问题

    with open('./datasets/train_data/train.csv','r',encoding='utf-8') as f_r:
        data = f_r.readlines()
        f_r.close()

    for line in data:
        line = line.strip()
        line = line.split('wenbenfengefu')
        text = line[0]
        spoList = line[1]
        spoList = json.loads(spoList)   #转为json格式的list
        if spoList == []:
            continue

        label_list = []
        re_ner = {}     #存储ner
        for spo in spoList:
            p = str(spo[1])     #关系，如[[[0, 28, 0, 36], 38, [6, 33, 6, 39]]]
            s_ids = spo[0]
            e_ids = spo[2]
            for i in range(len(s_ids)//2):
                s_start = s_ids[2*i]        #偶数位为头结点的开始位置
                o_start = s_ids[2*i + 1]    #奇数位为尾结点的开始位置
                s_end = e_ids[2*i]          #偶数位为头结点的结束位置
                o_end = e_ids[2*i + 1]      #奇数位为尾结点的结束位置
                if p not in label_list:
                    label_list.append(p)
                    start_ner_idx = []
                    end_ner_idx = []
                    start_ner_idx.append(str(s_start))
                    start_ner_idx.append(str(o_start))
                    end_ner_idx.append(str(s_end))
                    end_ner_idx.append(str(o_end))
                    re_ner[p] = [start_ner_idx,end_ner_idx]
                else:
                    start_ner_idx = re_ner[p][0]
                    end_ner_idx = re_ner[p][1]
                    start_ner_idx.append(str(s_start))
                    start_ner_idx.append(str(o_start))
                    end_ner_idx.append(str(s_end))
                    end_ner_idx.append(str(o_end))

        #保存关系训练样本
        label_string = ' '.join(label_list)     #一条样本中可能包含多个label
        relation_line = text + 'jjjj' + label_string + '\n'     #文本 + 关系标签
        f_w_relation.write(relation_line)

        #保存ner训练样本，一条样本中可能包含多个ner，对每个ner分别保存
        for k,v in re_ner.items():
            p_query = query_mapping[k]      #问题 字符串
            start_ner_idx = v[0]
            end_ner_idx = v[1]
            assert len(start_ner_idx) == len(end_ner_idx)
            #问题 + 文本 + ner开始位置 + ner结束位置
            ner_line = p_query + 'jjjj' + text + 'jjjj' + ' '.join(start_ner_idx) + 'jjjj' + ' '.join(end_ner_idx) + '\n'
            f_w_ner.write(ner_line)


# 生成用于关系训练的训练集合验证集
def create_relation_train_valid_sample():
    relation_data = './datasets/train_data/relation_datasets.csv'
    relation_train = './datasets/train_data/re/relation_datasets_train.csv'
    relation_valid = './datasets/train_data/re/relation_datasets_valid.csv'

    if not os.path.exists(relation_train):
        with open(relation_data, 'r', encoding='utf-8') as f:
            datasets = f.readlines()
            f.close()

        # 取前80%作为训练集
        f_train = open(relation_train, 'w', encoding='utf-8')
        train_dataset_len = int(len(datasets) * 0.8)
        train_dataset = datasets[:train_dataset_len]
        random.shuffle(train_dataset)
        for i in range(len(train_dataset)):
            f_train.write(train_dataset[i])

    if not os.path.exists(relation_valid):
        with open(relation_data, 'r', encoding='utf-8') as f:
            datasets = f.readlines()
            f.close()

        # 取后20%作为验证集
        f_valid = open(relation_valid, 'w', encoding='utf-8')
        valid_dataset = datasets[int(len(datasets) * 0.8):]
        random.shuffle(valid_dataset)
        for i in range(len(valid_dataset)):
            f_valid.write(valid_dataset[i])

# 生成用于实体训练的训练集合验证集
def create_ner_train_valid_sample():
    ner_data = './datasets/train_data/ner_datasets.csv'
    ner_train = './datasets/train_data/ner/ner_datasets_train.csv'
    ner_valid = './datasets/train_data/ner/ner_datasets_valid.csv'

    if not os.path.exists(ner_train):
        with open(ner_data, "r", encoding="utf-8") as f:
            data_sets = f.readlines()
            f.close()

        ner_train_len = int(len(data_sets) * 0.8)
        train_datasets = data_sets[:ner_train_len]
        random.shuffle(train_datasets)
        with open(ner_train, "w", encoding="utf-8") as f_train:
            for i in range(len(train_datasets)):
                f_train.write(train_datasets[i])
            f_train.close()

    if not os.path.exists(ner_valid):
        with open(ner_data, "r", encoding="utf-8") as f:
            data_sets = f.readlines()
            f.close()

        valid_datasets = data_sets[int(len(data_sets) * 0.8):]
        random.shuffle(valid_datasets)
        with open(ner_valid, "w", encoding="utf-8") as f_valid:
            for i in range(len(valid_datasets)):
                f_valid.write(valid_datasets[i])


if __name__ == "__main__":
    create_relation_ner_sample()
    create_relation_train_valid_sample()
    create_ner_train_valid_sample()
    print("success!")