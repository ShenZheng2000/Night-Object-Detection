import matplotlib
import matplotlib.pyplot as plt
from detectron2_gradcam import Detectron2GradCAM

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from gradcam import GradCAM, GradCamPlusPlus
import os
import torch

# Set a random seed for reproducibility
torch.manual_seed(0)

# If you are using CUDA, you should also set the seed for it
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # For multi-GPU.

# Additionally, you may want to disable hash-based algorithms for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


plt.rcParams["figure.figsize"] = (30,10)

# img_path = "img/input.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val/b1c9c847-3bda4659.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_debug/0a0c3694-4cc8b0e3.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day/0a5b3a25-81c3be4d.jpg" # => This is good, pick it!
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day/0a7cbfc6-bfeebf8d.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b2bceb54-4d3cc92c.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train/31686ca2-c70e8b39.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b1e1a7b8-65ec7612.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b3dd5345-15bbf8ed.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b5afd0ea-222fae7e.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b8f2d63d-6a944fa4.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b8c0457f-d7a1d2b7.jpg" # => This is OK
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b20eae11-6817ba7a.jpg"

# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b38b2b6a-cb374ce8.jpg" => This looks good
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b53cb17b-4df01599.jpg" => skip

# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b80fb9b6-b12f0c90.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b263f57d-1ccf6e51.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b275c028-6a0665cb.jpg"  => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b836b14a-fb13c905.jpg"  => skip

# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b892f130-90ed9d1f.jpg"  => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b953ef60-6c2cb9a1.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b4561bec-26a99096.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b24380ab-6dbeb908.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/b29026f0-d5a5ee8c.jpg" => skip

# img_path = '/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bbc9ec36-8e63f11e.jpg' => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bbfcd002-f8531a65.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bc140c47-76775eb4.jpg" => skip
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bcaf73c1-e0c7165a.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bcb441fc-a88da923.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bcf70f06-5bd41cdd.jpg" # => This looks good
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bd3bd3ab-79f6d287.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bd8f4dfa-52cf9375.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bd812175-fc557aa4.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/becccb13-74e09a25.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/befbf2cc-bec128ef.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_rainy/bf8ff5f5-877edaa0.jpg"

# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_day/b1cebfb7-284f5117.jpg"

# Find images that both satisfy warping and grad-cam requirements => single small object with good visualizations
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_debug/0a0c3694-9572f64f.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_debug/0a0ceca1-4148e482.jpg"
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_day/b1c66a42-6f7d68ca.jpg"
img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_day/b1d7b3ac-5744370e.jpg"

basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")

# # NOTE: this is for the warped image
# config_file = "/home/aghosh/Projects/2PCNet/Methods/Night-Object-Detection/configs/bdd100k_10_18_bbox.yaml"
# model_file = "/longdata/anurag_storage/2PCNet/outputs_11_14_det_ckpts/bdd100k_10_18_bbox/model_final.pth"

#model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")



layer_name = "backbone.res5.2.conv3"
# instance = 8 #CAM is generated per object instance, not per class!
# instance = 5 #CAM is generated per object instance, not per class!


def main():

    # TODO: try this once we have GPUs
    for model_name in [
                        # "bdd100k_10_18_baseline", 
                       "bdd100k_10_18_fovea",
                       "bdd100k_10_18_tpp",
                    #    "bdd100k_10_18_bbox"
                       ]:

        surfix = 'jpg' # for debug only
        # surfix = 'pdf'

        # NOTE: this is for the baseline
        config_file = f"/home/aghosh/Projects/2PCNet/Methods/Instance-Warp/Night-Object-Detection/configs/{model_name}.yaml"
        model_file = f"/longdata/anurag_storage/2PCNet/2PCNet/outputs/{model_name}/model_final.pth"

        config_list = [
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.5",
        "MODEL.ROI_HEADS.NUM_CLASSES", "10",
        "MODEL.WEIGHTS", model_file
        ]    

        cam_extractor = Detectron2GradCAM(config_file, config_list, img_path=img_path)
        grad_cam = GradCamPlusPlus

        for instance in range(0, 5): # range(0, xxx)
            image_dict, cam_orig = cam_extractor.get_cam(target_instance=instance, layer_name=layer_name, grad_cam_instance=grad_cam)
            # print("image_dict['output']['instances'] = ", image_dict["output"]["instances"])

            v = Visualizer(image_dict["image"], MetadataCatalog.get(cam_extractor.cfg.DATASETS.TRAIN[0]), scale=1.0)
            out = v.draw_instance_predictions(image_dict["output"]["instances"][instance].to("cpu"))
            
            plt.imshow(out.get_image(), interpolation='none')
            plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)
            # plt.title(f"CAM for Instance {instance} (class {image_dict['label']})")
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])

            output_folder = f"gradcam_output/{model_name}/{basename}"
            os.makedirs(output_folder, exist_ok=True)

            plt.savefig(f"{output_folder}/instance_{instance}_cam.{surfix}",
                        bbox_inches='tight', pad_inches=0) # , dpi=100
            plt.close()
            
            # TODO: save as pdf once all debug is DONE
        # plt.show()


if __name__ == "__main__":
    main()
