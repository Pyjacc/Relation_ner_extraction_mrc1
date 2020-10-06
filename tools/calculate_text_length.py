import matplotlib.pyplot as plt

datapath = '../datasets/train_data/train.csv'

with open(datapath,'r',encoding='utf-8') as f:
    data = f.readlines()
f.close()

lensentence = {}

for line in data:
    line = line.strip()
    line = line.split("wenbenfengefu")
    text = line[0]
    lens = len(text)
    if lens not in lensentence.keys():
        lensentence[lens] = 1
    else:
        lensentence[lens] += 1

plt.bar(lensentence.keys(),lensentence.values())
plt.show()

