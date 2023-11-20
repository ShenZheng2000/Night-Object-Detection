import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

# Sample data
groups = ['AP_small', 'AP_medium', 'AP_large']
original = [9.52, 26.01, 43.26]
ours = [9.84, 27.61, 46.62]

# Set the positions of the bars on the x-axis
x = np.arange(len(groups))

# Set the width of the bars
bar_width = 0.35

# Create subplots with broken y-axis
fig = plt.figure(figsize=(10, 5))
bax = brokenaxes(ylims=((0, 10), (25, 30), (40, 50)), hspace=.05)

# Set the positions of the bars on the x-axis with an offset to align properly
offset = bar_width / 2
bax.bar(x - offset, original, width=bar_width, label='Original', color='#e41a1c')
bax.bar(x + offset, ours, width=bar_width, label='Ours', color='#377eb8')

# Set labels and title
bax.set_xlabel('Categories', fontsize=14)
bax.set_ylabel('Values', fontsize=14)

# Set the positions and labels for the x-ticks
bax.set_xticks(x)
bax.set_xticklabels(groups)

# Set the legend with a larger font size
bax.legend(fontsize='large')

# Annotations on broken axes
for index in range(len(groups)):
    bax.text(x[index] - offset, original[index] + 0.1, f'{original[index]:.2f}', ha='center', va='bottom')
    bax.text(x[index] + offset, ours[index] + 0.1, f'{ours[index]:.2f}', ha='center', va='bottom')

# Apply tight layout to ensure everything fits without overlapping
plt.tight_layout()

# Save the figure
fig.savefig('comparison_bar_plot_broken_axis.png')