'''
对训练数据进行整理，生成包含关系和实体的训练数据集
注意问题：
（1）数据存在同个关系对应多个实体的情况
（2）一个样本存在多个关系
（3）一个关系下，存在多个实体
'''

import json

def load_label():
    p_label = {}    #字典，存储label，“关系”：“label” 形式
    with open('./datasets/relation_label.csv','r',encoding='utf-8') as f0:
        labels = f0.readlines()
        f0.close()

    for line in labels:
        line = line.strip()
        line = line.split(',')
        p_label[line[1]] = int(line[0])

    return p_label

# 在text中查询实体的开始位置和结束位置
def get_index(text,entity):
    text = str(text)            #文本
    entity = str(entity)        #实体
    entity_len = len(entity)
    start = text.index(entity)  #开始位置
    end = start + entity_len    #结束位置
    return start, end

# 读取train_data，构建样本
def build_samples(p_label):
    train_data = open('./datasets/train_data/train.csv', 'w', encoding='utf-8')  # 用于保存转换后的样本数据

    with open('./datasets/train_data.json','r',encoding='utf-8') as f_r:
        data = f_r.readlines()
        f_r.close()

    for line in data:
        line = json.loads(line.strip('\n')) #json格式读取方法（需要loads，不能直接读取。csv格式可以直接读取）
        text = line['text']             #文本
        spoList = line['spo_list']      #三元组信息
        p_list = {}     #存放每个样本包含的关系（一个样本中可能包含多种关系），形势为：“关系”：“开始位置，结束位置”

        for spo in spoList:
            p = spo['predicate']    #关系，predicate
            p = p_label[p]          #对应的label（将文字转换为对应的label）
            s = spo['subject']      #头结点，subject
            o = spo['object']['@value'] #尾结点，object
            s_s,s_e = get_index(text,s) #头结点开始和结束位置，s_s:subject_start,s_e:subject_end
            o_s,o_e = get_index(text,o) #尾结点开始和结束位置，o_s:object_start,o_e:object_end

            if p not in p_list.keys():
                start = []  #一个样本中所有实体的开始位置
                end = []    #一个样本中所有实体的结束位置
                start.append(s_s)
                start.append(o_s)
                end.append(s_e)
                end.append(o_e)
                p_list[p] = start, end
            else:
                start, end = p_list[p]
                start.append(s_s)
                start.append(o_s)
                end.append(s_e)
                end.append(o_e)

        # 将关系和实体处理成如下形势：[[头结点的开始位置，尾结点的开始位置]，关系，[头结点的结束位置，尾结点的结束位置]],如
        # [[[8,9],3,[9,10]],[[1,5],2,[3,6]]]
        final_label = []
        for k,v in p_list.items():
            s,e = p_list[k]
            temp = []
            temp.append(s)  #所有实体的开始位置
            temp.append(k)  #关系(label)
            temp.append(e)  #所有实体的结束位置
            final_label.append(temp)

        sample = text + 'wenbenfengefu'  + str(final_label) + '\n'  #构建样本
        train_data.write(sample)    #保存样本

    train_data.close()


if __name__ == "__main__":
    label = load_label()
    build_samples(label)