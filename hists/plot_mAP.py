

import numpy as np
import matplotlib.pyplot as plt

task = 'day2night' # [day, day2night, clear2rainy]

# Sample data
groups = ['Small', 'Medium', 'Large']

if task == 'day2night':
    original = [9.52, 26.01, 43.26]
    ours = [9.84, 27.61, 46.62]
elif task == 'day':
    original = [13.93, 37.58, 50.96]
    ours = [14.48, 38.40, 52.75]
elif task == 'clear2rainy':
    original = [10.88, 34.09, 48.99]
    ours = [11.04, 36.65, 49.00]

# Set the positions of the bars on the x-axis
x = np.arange(len(groups))

# Set the width of the bars
bar_width = 0.35

# Create the bar plot
# plt.bar(x - bar_width/2, original, width=bar_width, label='Original', color='#e41a1c')
# plt.bar(x + bar_width/2, ours, width=bar_width, label='Ours', color='#377eb8')

plt.bar(x - bar_width/2, original, width=bar_width, label='Original', color='#fb8072')
plt.bar(x + bar_width/2, ours, width=bar_width, label='Ours', color='#80b1d3')

# Set labels and title
plt.xlabel('Categories', fontsize=20)
plt.ylabel('mAP50', fontsize=20)

# Set the y-axis limit to better visualize the differences
plt.ylim(min(original+ours) - 5, max(original+ours) + 5)

# TODO: skip [10, 20] and [30, 40]

# Set the positions and labels for the x-ticks
plt.xticks(x, groups, fontsize=18)
# plt.yticks([0, 10, 20, 30, 40, 50,], fontsize=18)
plt.yticks(fontsize=18)

# Set the legend with a larger font size
plt.legend(fontsize=16)

# Apply tight layout to ensure everything fits without overlapping
plt.tight_layout()

for index in range(len(groups)):
    plt.text(x[index] - bar_width/2, original[index] + 0.3, f'{original[index]:.2f}', ha='center', fontsize=16)
    plt.text(x[index] + bar_width/2, ours[index] + 0.3, f'{ours[index]:.2f}', ha='center', fontsize=16)

# Show and save the plot
# plt.savefig(f'bar_plot_{task}.png') # for debug only
plt.savefig(f'bar_plot_{task}_2.pdf')