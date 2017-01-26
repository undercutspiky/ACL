import numpy as np
from sklearn.manifold import TSNE

drops = np.load("loss-drops.npy")
model = TSNE(n_components=2, verbose=2)

drops = np.transpose(drops,[1,0,2])
drops = drops.reshape(3130,313)
#np.set_printoptions(suppress=True)
embeddings = np.array(model.fit_transform(drops))
np.save("tsne-embeddings-10-epochs",embeddings)

