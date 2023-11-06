import numpy as np
import matplotlib.pyplot as plt

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

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Sqrt(Area)' if use_sqrt else 'Area')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(min_val, max_val)  # Set the x-axis limits
    plt.savefig(output_path)
    plt.close()


def load_data_from_txt(filename):
    return np.loadtxt(filename)

def main():
    grid_net_names = [
        'FixedKDEGrid',
        'CuboidGlobalKDEGrid',
        'PlainKDEGrid',
        'PlainKDEGrid_False_True', # l1 scale
        'PlainKDEGrid_True_False', # l2 scale
    ]

    hist_data_orig = load_data_from_txt("hist_data_orig.txt")

    for grid_net in grid_net_names:
        hist_data = load_data_from_txt(f"hist_data_{grid_net}.txt")
        plot_histogram(hist_data_orig, hist_data, 'Original BBoxes', grid_net, 
                       num_bins=50, title=f'Origin vs {grid_net}', output_path=f'Origin_vs_{grid_net}.png')

if __name__ == "__main__":
    main()
