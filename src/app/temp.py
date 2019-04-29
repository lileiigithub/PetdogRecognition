
label_name = []

with open("label_name.txt",encoding='utf-8') as file:
    for line in file.readlines():
        line = line.strip()
        name = (line.split('-')[-1])
        if name.count('|') > 0:
            name = name.split('|')[-1]
        print(name)
        label_name.append((name))

for item in label_name:
    with open('label_name_1.txt','a') as file:
        file.write(item+'\n')