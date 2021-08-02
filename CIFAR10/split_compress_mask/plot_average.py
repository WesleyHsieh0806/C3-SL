from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm
'''***************
* Plot the histogram of Average
******************
'''
Avg_path = os.path.join(
    os.path.dirname(__file__),
    'Average')

feature_bylabel = np.load(os.path.join(Avg_path, 'features_avg_Batch64.npy'))
gradient_bylabel = np.load(os.path.join(Avg_path, 'gradients_avg_Batch64.npy'))

print("Size of features average:{}".format(feature_bylabel.shape))
print("Size of gradients average:{}".format(gradient_bylabel.shape))


# Plot the histogram of Average for each label
for label in range(10):
    # Plot histogram
    plt.plot(range(1, feature_bylabel.shape[1]+1),
             feature_bylabel[label], color='r')
    plt.savefig(os.path.join(
        Avg_path, 'Avg_feature_label{}.png'.format(label)))
    plt.close()

    # plot histogram
    plt.plot(range(1, gradient_bylabel.shape[1]+1),
             gradient_bylabel[label], color='purple')
    plt.savefig(os.path.join(
        Avg_path, 'Avg_gradient_label{}.png'.format(label)))
    plt.close()

# Plot TSNE to check whether the Average of each label is similar
color = cm.rainbow(np.linspace(0, 1, 11))
features_embedded = TSNE(n_components=2).fit_transform(feature_bylabel)

for i in range(10):
    plt.scatter(features_embedded[i, 0],
                features_embedded[i, 1], color=color[i], label='Avg of Label {}'.format(i))
plt.legend()
plt.savefig(os.path.join(
    Avg_path, 'Avg_TSNE.png'))
plt.close()
