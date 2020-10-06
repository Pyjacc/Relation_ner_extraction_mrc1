# Relation_ner_extraction_mrc1
关系实体抽取

### 1. data_clean.py中对训练数据进行整理，生成包含关系和实体的训练数据集 
### 2. data_process.py中生成实体训练的数据集 
### 3. relation_extraction.py中抽取关系 
### 4. entity_extraction_under_relation.py中进行实体抽取
### 5. 代码执行顺序如下：
#### 1）第一步：先执行data_clean.py，对数据进行整理
#### 2）第二步：执行data_process.py，生成关系训练集，验证集。实体训练集，验证集
#### 3）第三步：执行relation_extraction.py
### 6. 说明：
#### 1）工程基于pt20200816
#### 2）完整模型训练参考工程pt20200809
