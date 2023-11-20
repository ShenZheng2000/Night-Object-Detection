import numpy as np
import matplotlib.pyplot as plt

# NOTE: old three split function
def calculate_ratios(data, small_threshold, medium_threshold):
    small_ratio = len(data[data <= small_threshold]) / len(data)
    medium_ratio = len(data[(data > small_threshold) & (data <= medium_threshold)]) / len(data)
    large_ratio = len(data[data > medium_threshold]) / len(data)
    return small_ratio, medium_ratio, large_ratio


def plot_histogram(data1, data2, label1, label2, num_bins=50, title='', output_path='', use_sqrt=False):
    # # If use_sqrt is True, transform the data using square root
    # if use_sqrt:
    #     data1 = np.sqrt(data1)
    #     data2 = np.sqrt(data2)

    # Get the minimum and maximum values across both data sets
    min_val = min(data1.min(), data2.min())
    max_val = max(data1.max(), data2.max())

    # Define the bin edges for logarithmic scale
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)

    # Plot the histograms
    plt.hist(data1, bins=bin_edges, label=label1, color="#e41a1c", alpha=0.9)
    plt.hist(data2, bins=bin_edges, label=label2, color="#377eb8", alpha=0.9)

    # Add vertical dotted lines at 32^2 and 96^2 for small and medium size thresholds
    plt.axvline(x=32**2, color='#984ea3', linestyle='--', linewidth=4, label='Small/Medium thres')
    plt.axvline(x=96**2, color='#f781bf', linestyle='--', linewidth=4, label='Medium/Large thres')

    # # Annotating the lines
    # plt.text(32**2, plt.ylim()[1]*0.9, 'Small', color='green', ha='right')
    # plt.text(96**2, plt.ylim()[1]*0.9, 'Medium', color='purple', ha='right')
    # plt.text(96**2, plt.ylim()[1]*0.9, 'Large', color='purple', ha='left')

    plt.legend(loc='upper right', fontsize='large')
    # plt.title(title, fontsize=16)
    plt.xlabel('Area', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xscale('log')
    plt.xlim(min_val, max_val)  # Set the x-axis limits
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path)

    print("saveed to ", output_path)

    plt.close()


def load_data_from_txt(filename):
    return np.loadtxt(filename)

def main():
    grid_net_names = [
        # 'FixedKDEGrid',
        # 'CuboidGlobalKDEGrid',
        'PlainKDEGrid',
    ]

    # NOTE: add tod here
    # tod = 'night'

    dataset = 'bdd100k'
    # dataset = 'cityscapes'
    # dataset = 'acdc'

    warp_scale = 1.0
    surfix = 'pdf' # png or pdf

    hist_data_orig = load_data_from_txt(f"hist_data_{dataset}_orig.txt")

    for grid_net in grid_net_names:
        hist_data = load_data_from_txt(f"hist_data_{dataset}_{grid_net}.txt")

        hist_data = hist_data * warp_scale

        grid_net_save = 'Ours' if grid_net == 'PlainKDEGrid' else grid_net

        plot_histogram(hist_data_orig, hist_data, 'Original', grid_net_save, 
                       num_bins=50, 
                    #    title=f'Original vs {grid_net_save} ({warp_scale}x)',
                        output_path=f'{grid_net_save}_{warp_scale}_{dataset}.{surfix}')
        
        # # Calculate and print the ratios for hist_data_orig
        # small_ratio_orig, medium_ratio_orig, large_ratio_orig = calculate_ratios(hist_data_orig, 32**2, 96**2)
        # print(f'Original Ratios =================> Small: {small_ratio_orig:.2%}, Medium: {medium_ratio_orig:.2%}, Large: {large_ratio_orig:.2%}')
        
        # # Calculate and print the ratios for hist_data
        # small_ratio, medium_ratio, large_ratio = calculate_ratios(hist_data, 32**2, 96**2)
        # print(f'{grid_net_save} Ratios =================> Small: {small_ratio:.2%}, Medium: {medium_ratio:.2%}, Large: {large_ratio:.2%}')

if __name__ == "__main__":
    main()
