import numpy as np
import matplotlib.pyplot as plt
import math
from get_count import calculate_ratios, cal_std, load_data_from_txt, calculate_score


grid_net = 'PlainKDEGrid'

# dataset = 'bdd100k'
dataset = 'cityscapes'
# dataset = 'acdc'

category = 'car'

if dataset == 'bdd100k':
    img_width = 1280
    img_height = 720    
else:
    img_width = 2048
    img_height = 1024


def get_norm(data):
    # divide first by width, then by height
    normalized_data = data.copy()

    print("Before, max width is", max(normalized_data[:, 0]))
    print("Before, max height is", max(normalized_data[:, 1]))

    normalized_data[:, 0] = normalized_data[:, 0] / img_width
    normalized_data[:, 1] = normalized_data[:, 1] / img_height

    print("After, max width is", max(normalized_data[:, 0]))
    print("After, max height is", max(normalized_data[:, 1]))

    return normalized_data


def compute_mean_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def main():

    hist_data_orig = load_data_from_txt(f"hist_data_{dataset}_orig_hw_{category}.txt")
    hist_data = load_data_from_txt(f"hist_data_{dataset}_{grid_net}_hw_{category}.txt")

    # Normalize the data
    normalized_orig_data = get_norm(hist_data_orig)
    normalized_data = get_norm(hist_data)

    # Compute mean and std for original and model data
    mean_orig, std_orig = compute_mean_std(normalized_orig_data)
    mean_model, std_model = compute_mean_std(normalized_data)
    
    # Create two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot for original data
    axs[0].scatter(normalized_orig_data[:, 0], normalized_orig_data[:, 1],
                    label='Original', color='#e41a1c', alpha=1)
    axs[0].set_xlabel('Normalized Width', fontsize=20)
    axs[0].set_ylabel('Normalized Height', fontsize=20)
    axs[0].set_title('Original', fontsize=20)

    # Scatter plot for model data
    axs[1].scatter(normalized_data[:, 0], normalized_data[:, 1],
                    label=f'Ours', color='#377eb8', alpha=1)
    axs[1].set_xlabel('Normalized Width', fontsize=20)
    axs[1].set_ylabel('Normalized Height', fontsize=20)
    axs[1].set_title(f'Ours', fontsize=20)

    # # Add legend
    # axs[0].legend()
    # axs[1].legend()

    # Set x-axis and y-axis tick label size for both subplots
    for ax in axs:
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Adjust spacing
    plt.tight_layout()

    # Save the figure as one PNG file
    plt.savefig(f"scatter_plots_{dataset}_{category}.png")

    # Print mean and std
    print(f"Original Data - Mean Width: \
          {mean_orig[0]:.2f}, Mean Height: {mean_orig[1]:.2f}, \
          Std Width: {std_orig[0]:.2f}, Std Height: {std_orig[1]:.2f}")
    
    print(f"Model Data - Mean Width: \
          {mean_model[0]:.2f}, Mean Height: {mean_model[1]:.2f}, \
          Std Width: {std_model[0]:.2f}, Std Height: {std_model[1]:.2f}")


if __name__ == "__main__":
    main()
