import numpy as np
import scipy
from sklearn.manifold import TSNE

drops = np.load("loss-drops.npy")
model = TSNE(n_components=2, verbose=2)

drops = np.transpose(drops,[1,0,2])
drops -= np.mean(drops, axis=0, keepdims=True) 
drops /= np.std(drops, axis=(0,2), keepdims=True)
drops = drops.reshape(313*20,313)
#drops = scipy.sign(drops)
print drops
#np.set_printoptions(suppress=True)
embeddings = np.array(model.fit_transform(drops))
np.save("tsne-mean_std",embeddings)

