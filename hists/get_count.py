import numpy as np
import matplotlib.pyplot as plt
import math


# NOTE: old three split function
def calculate_ratios(data, thresholds):
    ratios = []
    prev_threshold = -float('inf')
    
    for threshold in thresholds:
        ratio = len(data[(data > prev_threshold) & (data <= threshold)]) / len(data)
        ratios.append(ratio)
        prev_threshold = threshold
    
    # Calculate the ratio for values greater than the last threshold
    ratio = len(data[data > thresholds[-1]]) / len(data)
    ratios.append(ratio)
    
    return tuple(ratios)


def cal_std(data):
    return np.std(data)

def load_data_from_txt(filename):
    return np.loadtxt(filename)

def main():
    grid_net_names = [
        # 'FixedKDEGrid',
        # 'CuboidGlobalKDEGrid',
        'PlainKDEGrid',
    ]

    # dataset = 'bdd100k'
    dataset = 'cityscapes'
    # dataset = 'acdc'

    # Define the thresholds
    # thresholds = [32**2, 96**2] # three-way split => 4
    # thresholds = [32**2, 64**2, 96**2] # four-way split => 4
    # thresholds = [32**2, 43**2, 64**2, 96**2] # five-way split  => 4
    # thresholds = [32**2, 48**2, 64**2, 80**2, 96**2] # six-way split => 8 (changes for bdd100k), 4 (same for cityscapes)

    warp_scale = 1.0

    hist_data_orig = load_data_from_txt(f"hist_data_{dataset}_orig.txt")

    for grid_net in grid_net_names:
        hist_data = load_data_from_txt(f"hist_data_{dataset}_{grid_net}.txt")

        hist_data = hist_data * warp_scale
        
        # # Calculate and print the ratios for hist_data_orig
        # ratios_orig = calculate_ratios(hist_data_orig, thresholds)
        # scores_orig = calculate_score(ratios_orig)
        # formatted_ratios_orig = [f'{ratio * 100:.2f}%' for ratio in ratios_orig]
        # print(f'Original Ratios =================> {formatted_ratios_orig} scores: {scores_orig}')

        # # Calculate and print the ratios for hist_data
        # ratios = calculate_ratios(hist_data, thresholds)
        # scores = calculate_score(ratios)
        # formatted_ratios = [f'{ratio * 100:.2f}%' for ratio in ratios]
        # print(f'Our Ratios =================> {formatted_ratios}')

        # hist_data_orig
        variance_orig = cal_std(hist_data_orig)

        # Calculate and print the variance of hist_data
        variance = cal_std(hist_data)

        # Divide all numbers by 1000 when printing
        print(f"Mean of hist_data_orig: {hist_data_orig.mean() / 1000:.2f}")
        print(f'Std of hist_data_orig: {variance_orig / 1000:.2f}')
        print(f"Mean of hist_data: {hist_data.mean() / 1000:.2f}")
        print(f'Std of hist_data: {variance / 1000:.2f}')


def calculate_score(tuple_values):
    temp = 0

    for i in range(len(tuple_values) - 1):
        cur = tuple_values[i] / tuple_values[i + 1]
        # print("cur is", cur)
        temp += cur

    score = 2 ** max(math.floor(math.log2(temp)), 0)
    
    return score


if __name__ == "__main__":
    main()
