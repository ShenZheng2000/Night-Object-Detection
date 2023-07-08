# Purpose: render reflection based on a single light souce
# Later: consider reflection due to multiple light sources 
# Later: consider refleciton dues to road surface unevenness (i.e. not a flat mirror plane)

# elevation (no need for now): 
    # head/tail-light: 0.6-0.9m
    # camera: 1.2-1.5m

# Decide reflection area
    # light source coords = (x1, y1), obtained from light keypoints
    # reflect coords = (x1, y2)
        # y2 selected randomly from wheel keypoints to image bottom

# Fill reflection area
    # similar to gaussian heatmap
    # area rect instead of square

# Think: do headlight or taillight contribute to reflection
