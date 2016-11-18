import numpy as np
import cPickle as cp
from scipy.io import loadmat

def save_split(name, obj):
    f = open(name,"wb")
    cp.dump(obj,f,cp.HIGHEST_PROTOCOL)
    f.close()

data = loadmat("train_32x32.mat")
x = data['X']
y = data['y']

x = np.transpose(x,axes=[3,0,1,2])
y = np.array([i[0] for i in y])
y[np.where(y==10)[0]] = 0
y = np.array(y)

cv = []
for i in xrange(10):
    l = len(np.where(y==i)[0])
    cv.append(np.random.randint(l, size=int(0.2*l)))
    
cv = np.hstack(cv)
valid_x = x[cv,:,:,:]
valid_y = y[cv]
train_x = np.delete(x,cv,axis=0)
train_y = np.delete(y,cv)
train_x_1 = train_x[:32000,:,:,:]
train_x_2 = train_x[32000:,:,:,:]

save_split('train_x_1',train_x_1)
save_split('train_x_2',train_x_2)
save_split('valid_x',valid_x)
save_split('valid_y',valid_y)
save_split('train_y',train_y)
