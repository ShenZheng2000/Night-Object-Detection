# from .invert_grid import invert_grid
# # NOTE: include vp oob conditions at last
# from .fovea import make_warp_aug, simple_test
# from .fixed_grid import FixedGrid
# from .path_blur import is_out_of_bounds, get_vanising_points
# from .night_aug import NightAug

# import torch.nn.functional as F

# TODO: write two function, one before backbone, and one after backbone
# Each function should be done end-to-end. 


# def zoom_lzu(label_data):
#     # for imgs in label_data:

#     #     my_shape = img.shape[1:]

#     #     fix_grid = FixedGrid(separable=True, 
#     #                         anti_crop=True, 
#     #                         input_shape=my_shape, 
#     #                         output_shape=my_shape)

#     #     grid, warped_imgs = simple_test(grid_generator, imgs, vanishing_point)

#     #     for i in label_data:
#     #         img = label_data[i]
#     #         # TODO: import make_warp_aug, get_vanising_points, and is_out_of_bounds
#     #         warped_imgs = make_warp_aug(img, vanishing_point, grid, use_ins=False)

#     # return warped_imgs
#     pass



# def unzoom_lzu(features):
#     # # TODO: define grid
#     # warped_x = features
#     # x = []
#     # # precompute and cache inverses
#     # separable = grid_generator.separable
#     # if inverse_grids is None:
#     #     inverse_grids = []
#     #     for i in range(len(warped_x)):
#     #         input_shape = warped_x[i].shape
#     #         inverse_grid = invert_grid(grid, input_shape,
#     #                                 separable=separable)[0:1]
#     #         inverse_grids.append(inverse_grid)

#     # # perform unzoom
#     # for i in range(len(warped_x)):
#     #     B = len(warped_x[i])
#     #     inverse_grid = inverse_grids[i].expand(B, -1, -1, -1)
#     #     unwarped_x = F.grid_sample(
#     #         warped_x[i], inverse_grid, mode='bilinear',
#     #         align_corners=True, padding_mode='zeros'
#     #     )
#     #     x.append(unwarped_x)

#     # print("x is", x)
#     # print("tuple(x) is", tuple(x))
#     pass