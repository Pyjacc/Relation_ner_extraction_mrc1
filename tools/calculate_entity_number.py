
import json

# relation_entity_mapping  = json.load(open('./datasets/53_schemas.json','r',encoding='utf-8'))
with open("../datasets/53_schemas.json",'r',encoding='utf-8') as f:
    data = f.readlines()

name = set()

for line in data:
    line = json.loads(line.strip('\n'))
    # print(line)
    s = line["subject_type"]
    o = line["object_type"]
    name.add(s)
    name.add(o)
name = list(name)
print(name)


