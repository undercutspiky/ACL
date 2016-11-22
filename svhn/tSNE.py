import numpy as np
from sklearn.manifold import TSNE

losses = np.load("losses-all-conv-2.npy")
model = TSNE(n_components=2, verbose=2)
losses = losses[:10,:]
losses = np.transpose(losses)
losses = losses[:60000,:]
#np.set_printoptions(suppress=True)
embeddings = np.array(model.fit_transform(losses))
np.save("tsne-embeddings-10-60K",embeddings)
