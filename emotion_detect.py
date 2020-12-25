import os
import nltk
import csv
import random
import numpy as np
#0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
#1 - the text of the tweet (Lyx is cool)

#Data
CDir = os.getcwd()
data = [[],[],[]]
#idx = {'0':0,'2':0,'4':0}
with open(CDir + "\\data.csv", newline='') as f:
    reader = csv.reader(f)
    for ls in reader:
        tp = int(ls[0])
        text = set(word.lower() for word in nltk.word_tokenize(ls[1]) if any(ch.isalpha() for ch in word))
        if tp== 0:
            data[0].append((tp,text))
        elif tp == 2:
            data[1].append((tp,text))
        elif tp == 4:
            data[2].append((tp,text)) 
        

#Separate training and testing data percentage : 10%
testdata = []
traindata = []
for ls in data: 
    num = len(ls)
    rand = random.choices(range(num),k=num//10)
    rand2 = list(set(range(num)).difference(set(rand)))
    testdata.extend([ls[idx] for idx in rand])
    traindata.extend([ls[idx] for idx in rand2])

features = set()
#parse sentance to extract features
for item in traindata:
    features.update(item[1])

datatb = []
for item in traindata:
    row = [int(word in item[1]) for word in features]
    row.append(item[0])
    datatb.append(row)

datatb_ = np.array(datatb)

#Create models using Kmeans


#test

print("end")
        
