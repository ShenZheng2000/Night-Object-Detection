

import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

# Sample data
groups = ['Small', 'Medium', 'Large']
original = [55.44, 32.11, 12.45]
ours = [28.17, 49.43, 22.40]


# Set the positions of the bars on the x-axis
x = np.arange(len(groups))

# Set the width of the bars
bar_width = 0.35

# Create the bar plot
plt.bar(x - bar_width/2, original, width=bar_width, label='Original', color='#e41a1c')
plt.bar(x + bar_width/2, ours, width=bar_width, label='Ours', color='#377eb8')

# Set labels and title
plt.xlabel('Categories', fontsize=20)
plt.ylabel('Percent', fontsize=20)

# Set the y-axis limit to better visualize the differences
plt.ylim(min(original+ours) - 5, max(original+ours) + 5)

# TODO: skip [10, 20] and [30, 40]

# Set the positions and labels for the x-ticks
plt.xticks(x, groups, fontsize=18)
plt.yticks(fontsize=18)

# Set the legend with a larger font size
plt.legend(fontsize=16)

# Apply tight layout to ensure everything fits without overlapping
plt.tight_layout()

for index in range(len(groups)):
    plt.text(x[index] - bar_width/2, original[index] + 0.3, f'{original[index]:.2f}', ha='center', fontsize=16)
    plt.text(x[index] + bar_width/2, ours[index] + 0.3, f'{ours[index]:.2f}', ha='center', fontsize=16)

# Show and save the plot
# plt.savefig('bar_plot_dist.png') # for debug only
plt.savefig('bar_plot_dist.pdf')