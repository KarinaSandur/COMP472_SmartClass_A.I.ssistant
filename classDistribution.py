import numpy as np
import matplotlib.pyplot as mpl

# Initialize data in bar graph
num_angry = 687
num_angry_train = (num_angry * 0.75)
num_angry_test = (num_angry * 0.15)
num_angry_val = (num_angry * 0.15)

num_focused = 514
num_focused_train = (num_focused * 0.75)
num_focused_test = (num_focused * 0.15)
num_focused_val = (num_focused * 0.15)

num_happy = 540
num_happy_train = (num_happy * 0.75)
num_happy_test = (num_happy * 0.15)
num_happy_val = (num_happy * 0.15)

num_neutral = 582
num_neutral_train = (num_neutral * 0.75)
num_neutral_test = (num_neutral * 0.15)
num_neutral_val = (num_neutral * 0.15)


# Initialize variables
classes = ['Angry', 'Focused', 'Happy', 'Neutral']
num_train_pics = [num_angry_train, num_focused_train, num_happy_train, num_neutral_train]
num_val_pics = [num_angry_val, num_focused_val, num_happy_val, num_neutral_val]
num_test_pics = [num_angry_test, num_focused_test, num_happy_test, num_neutral_test]

# Plot class distribution
x = np.arange(len(classes))
width = 0.30

fig, ax = mpl.subplots()
bar1 = ax.bar(x - width , num_train_pics, width, label='Training')
bar2 = ax.bar(x, num_val_pics, width, label="Validation")
bar3 = ax.bar(x + width, num_test_pics, width, label="Testing")

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_title('Class Distribution')
ax.set_ylabel('Number of Images')
ax.legend()

# Display class distribution
mpl.show()