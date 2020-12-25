import os,nltk,csv,random
import numpy as np
from scipy.spatial.distance import cdist

def CalculateAcc(new_lb,real_lb):
    acc = 0
    for idx in range(new_lb.shape[0]):
        if new_lb[idx] == real_lb[idx]:
            acc += 1
    acc = acc/new_lb.shape[0]
    return acc

def Kmeans(datatb_,K,real_lb):
    #Choose K center
    centers = [datatb_[np.random.choice(datatb_.shape[0], K, replace=False)]]

    labels=[]

    while True:
        #Calculate distance then get label of nearest neighbour
        dist = cdist(datatb_,centers[-1])
        labels = np.argmin(dist, axis = 1)

        #new centers
        new_centers = np.zeros((K, datatb_.shape[1]))
        for idx in range(K):
            # collect all points assigned to the idx-th cluster 
            clus = datatb_[labels == idx, :]
            # take average
            if clus.shape[0] != 0:
                new_centers[idx,:] = np.mean(clus, axis = 0)
            else:
                new_centers[idx,:] = centers[-1][idx,:]
        # Check if centers set is unchanged
        if (set([tuple(a) for a in centers[-1]]) == 
            set([tuple(a) for a in new_centers])):
            break
        centers.append(new_centers)

    #change labels due to correct label
    pos = np.array(range(labels.shape[0]))
    clus_lb1 = []
    clus_lb2 = []
    for idx in range(K):
        clus_lb1.append(set(pos[labels==idx]))
    for idx in range(3):
        clus_lb2.append(set(pos[real_lb==idx]))

    cr = np.zeros([K,3])
    new_lb = np.zeros([labels.shape[0],1])
    for idx in range(K):
        cr[idx,0] = len(clus_lb1[idx].intersection(clus_lb2[0]))
        cr[idx,1] = len(clus_lb1[idx].intersection(clus_lb2[1]))
        cr[idx,2] = len(clus_lb1[idx].intersection(clus_lb2[2]))

    id_ = np.argmax(cr, axis = 1)
    for idx in range(K):
        new_lb[labels == idx] = int(id_[idx])

    #check accuracy of training data:
    acc = CalculateAcc(new_lb,real_lb)
    #print(acc)
    return (centers[-1], new_lb, acc, id_, K)

#main
#0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
#1 - the text of the tweet (Lyx is cool)
#Data
CDir = os.getcwd()
data = [[],[],[]]
num_d = []
with open(CDir + "\\data.csv", newline='') as f:
    reader = csv.reader(f)
    for ls in reader:
        tp = int(ls[0])
        text = set(word.lower() for word in nltk.word_tokenize(ls[1]) if any(ch.isalpha() for ch in word))
        if tp == 0:
            data[0].append((0,text))
        elif tp == 2:
            data[1].append((1,text))
        elif tp == 4:
            data[2].append((2,text)) 
        

#Separate training and testing data percentage : 10%
testdata = []
traindata = []
for ls in data: 
    num = len(ls)
    rand = np.random.choice(num, num//10, replace=False)
    rand2 = list(set(range(num)).difference(set(rand)))
    num_d.append(len(rand2))
    testdata.extend([ls[idx] for idx in rand])
    traindata.extend([ls[idx] for idx in rand2])

features = set()
#parse sentance to extract features
for item in traindata:
    features.update(item[1])

datatb = []
real_lb = []
for item in traindata:
    row = [float(word in item[1]) for word in features]
    real_lb.append(item[0])
    datatb.append(row)

datatb_ = np.array(datatb)
#normalize data
#mean = np.mean(datatb_,axis=0)
#sta = np.std(datatb_, axis=0)

#for idx in range(datatb_.shape[1]):
#    datatb_[:,idx] -= mean[idx]
#    datatb_[:,idx] *= 2
#    datatb_[:,idx] /= sta[idx]

real_lb = np.array(real_lb)
#Create models using Kmeans
centers = []
lbs = []
accs = np.zeros([13])
id_=[]
ks=0
for K in range(3,16):
    centers.append([])
    lbs.append([])
    id_.append([])
    for idx in range(5):
        center,lb,acc,id1,_ = Kmeans(datatb_,K,real_lb)
        if acc > accs[K-3]:
            centers[K-3] = center
            lbs[K-3] = lb
            accs[K-3]=acc
            id_[K-3]=id1

#get best acc
for K in range(3,16):
    print(accs[K-3])
    acc = 0
    if acc < accs[K-3]:
        ks=K
print("\n")

'''
0.4074074074074074
0.45014245014245013
0.49002849002849
0.5071225071225072
0.42450142450142453
0.4700854700854701
0.48717948717948717
0.49002849002849
0.5042735042735043
0.49002849002849
0.50997150997151
0.5042735042735043
0.5185185185185185
'''

#store models
with open(CDir + "\\model.data","w") as f:
    f.write(str(ks) + '\n')
    f.write(" ".join(features)  + '\n')
    for idx in range(ks):
        f.write(" ".join(['{:.9f}'.format(x) for x in centers[ks-3][idx,:]])  + '\n')
    f.write(" ".join([ '{:d}'.format(x) for x in id_[ks-3]]) + '\n')
    f.close()

'''
CDir = os.getcwd()
centers=[]
idTr_=[]
with open(CDir + "\\model.data","r", newline='') as f:
    ks = int(f.readline())
    feat_ = set([item for item in f.readline().split(" ")])
    for idx in range(ks):
        centers.append([float(item) for item in f.readline().split(" ")])
    idTr_.extend([int(item) for item in f.readline().split(" ")])
    
'''
#test
datats = []
rl_testlb = []
for item in testdata:
    row = [float(word in item[1]) for word in features]
    rl_testlb.append(item[0])
    datats.append(row)

datats_ = np.array(datats)
rl_testlb = np.array(rl_testlb)
center = np.array(centers[ks-3])

dist = cdist(datats_,center)
test_lb = np.argmin(dist,axis = 1)
new_lb = np.zeros([test_lb.shape[0],1])
for idx in range(ks):
    new_lb[test_lb == idx] = id_[ks-3][idx]
print(CalculateAcc(new_lb,rl_testlb))

confusionmatrix = np.zeros([3,3])

for idx in range(new_lb.shape[0]):
    confusionmatrix[rl_testlb[idx],int(new_lb[idx,0])] +=1

for idx in range(3):
    print('{:.0f}  {:.0f}  {:.0f}\n'.format(confusionmatrix[idx,0],confusionmatrix[idx,1],confusionmatrix[idx,2]))

'''
datats = []
rl_testlb = []
for item in testdata:
    row = [float(word in item[1]) for word in feat_]
    rl_testlb.append(item[0])
    datats.append(row)

datats_ = np.array(datats)
rl_testlb = np.array(rl_testlb)
center = np.array(centers)

dist = cdist(datats_,center)
test_lb = np.argmin(dist,axis = 1)
new_lb = np.zeros([test_lb.shape[0],1])
for idx in range(ks):
    new_lb[test_lb == idx] = idTr_[idx]
print(CalculateAcc(new_lb,rl_testlb))
'''
print("end")
