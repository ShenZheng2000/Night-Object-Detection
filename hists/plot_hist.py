import numpy as np
import matplotlib.pyplot as plt

# TODO: add dotted vertical line for small, medium, large

def plot_histogram(data1, data2, label1, label2, num_bins=50, title='', output_path='', use_sqrt=False):
    # If use_sqrt is True, transform the data using square root
    if use_sqrt:
        data1 = np.sqrt(data1)
        data2 = np.sqrt(data2)

    # Get the minimum and maximum values across both data sets
    min_val = min(data1.min(), data2.min())
    max_val = max(data1.max(), data2.max())

    # Define the bin edges for logarithmic scale
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)

    # Plot the histograms
    plt.hist(data1, bins=bin_edges, label=label1, color='red', alpha=0.5)
    plt.hist(data2, bins=bin_edges, label=label2, color='blue', alpha=0.5)

    # Add vertical dotted lines at 32^2 and 96^2 for small and medium size thresholds
    plt.axvline(x=32**2, color='orange', linestyle='--', linewidth=1, label='Small/Medium threshold')
    plt.axvline(x=96**2, color='green', linestyle='--', linewidth=1, label='Medium/Large threshold')

    # # Annotating the lines
    # plt.text(32**2, plt.ylim()[1]*0.9, 'Small', color='green', ha='right')
    # plt.text(96**2, plt.ylim()[1]*0.9, 'Medium', color='purple', ha='right')
    # plt.text(96**2, plt.ylim()[1]*0.9, 'Large', color='purple', ha='left')

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Sqrt(Area)' if use_sqrt else 'Area')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(min_val, max_val)  # Set the x-axis limits
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_data_from_txt(filename):
    return np.loadtxt(filename)

def main():
    grid_net_names = [
        'FixedKDEGrid',
        'CuboidGlobalKDEGrid',
        'PlainKDEGrid_False_False',
        # 'PlainKDEGrid_False_True', # l1 scale
        # 'PlainKDEGrid_True_False', # l2 scale
    ]

    # NOTE: add tod here
    tod = 'night'

    hist_data_orig = load_data_from_txt(f"hist_data_{tod}_orig.txt")

    for grid_net in grid_net_names:
        hist_data = load_data_from_txt(f"hist_data_{tod}_{grid_net}.txt")
        plot_histogram(hist_data_orig, hist_data, 'Original BBoxes', grid_net, 
                       num_bins=50, title=f'Origin vs {tod}_{grid_net}', output_path=f'Origin_vs_{tod}_{grid_net}.png')

if __name__ == "__main__":
    main()
