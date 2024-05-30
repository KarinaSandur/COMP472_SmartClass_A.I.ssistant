import numpy as np
import matplotlib.pyplot as mpl

classes = ['Neutral', 'Angry', 'Happy', 'Focused']
num_train_pics = [421, 442, 405, 403]
num_test_pics = [138, 228, 111, 101]

x = np.arange(len(classes))
width = 0.30

fig, ax = mpl.subplots()
bar1 = ax.bar(x - width / 2, num_train_pics, width, label='Training')
bar2 = ax.bar(x + width / 2, num_test_pics, width, label="Testing")

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_title('Class Distribution')
ax.set_ylabel('Number of Images')
ax.legend()

mpl.show()