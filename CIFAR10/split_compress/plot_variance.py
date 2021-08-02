from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm
'''***************
* Plot the histogram of variance
******************
'''
Var_path = os.path.join(
    os.path.dirname(__file__),
    'Variance')

labels = np.load(os.path.join(Var_path, 'labels_Batch64.npy'))
features = np.load(os.path.join(Var_path, 'features_Batch64.npy'))
gradients = np.load(os.path.join(Var_path, 'gradients_Batch64.npy'))

print("Size of labels:{}".format(labels.shape))
print("Size of features:{}".format(features.shape))
print("Size of gradients:{}".format(gradients.shape))


# Plot the histogram of variance for each label
nof_label = np.max(labels)+1
start = np.min(labels)
feature_bylabel = []
gradient_bylabel = []
for label in range(start, nof_label):
    # Label mask
    label_mask = (labels == label)
    feature_bylabel.append(np.var(features[label_mask], axis=0, keepdims=True))
    gradient_bylabel.append(
        np.var(gradients[label_mask], axis=0, keepdims=True))

    # Plot histogram
    plt.plot(range(1, features[label_mask].shape[1]+1),
             np.var(features[label_mask], axis=0), color='r')
    plt.savefig(os.path.join(
        Var_path, 'Var_feature_label{}.png'.format(label)))
    plt.close()

    # plot histogram
    plt.plot(range(1, gradients[label_mask].shape[1]+1), np.var(
        gradients[label_mask], axis=0), color='purple')
    plt.savefig(os.path.join(
        Var_path, 'Var_gradient_label{}.png'.format(label)))
    plt.close()

# Plot TSNE to check whether the variance of each label is similar
color = cm.rainbow(np.linspace(start, 1, nof_label))
features = np.concatenate(feature_bylabel, axis=0)
features_embedded = TSNE(n_components=2).fit_transform(features)

for i in range(start, nof_label):
    plt.scatter(features_embedded[i, 0],
                features_embedded[i, 1], color=color[i], label='Var of Label {}'.format(i))
plt.legend()
plt.savefig(os.path.join(
    Var_path, 'Var_TSNE.png'))
plt.close()
