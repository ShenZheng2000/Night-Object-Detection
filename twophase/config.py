from detectron2.config import CfgNode as CN


def add_teacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL_DEPTH = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "studentteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.SCALE_STEPS=(0,)
    _C.SEMISUPNET.SCALE_LIST=(1.0,)

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True

    _C.NIGHTAUG = True
    _C.STUDENT_SCALE = True
    _C.CONSISTENCY = True
    _C.ONE_STAGE = False
    _C.USE_DEPTH = False
    _C.USE_VIS = False

    # NOTE: adapted from MIC: https://github.com/lhoyer/MIC/blob/81782e070f5e9f1911c9eeaa0353129b4f073850/det/maskrcnn_benchmark/config/defaults.py#L38
    _C.USE_MASK = False
    _C.USE_MASK_SRC = False
    _C.MASKING_SPA = False
    _C.MASKING_CONS = False
    _C.MASKING_BLOCK_SIZE = 32
    _C.MASKING_RATIO = 0.3
    _C.MASKING_AUGMENTATION = False
    _C.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    _C.PIXEL_STD = [1., 1., 1.]


    # NOTE: add motion blur
    _C.MOTION_BLUR = False
    _C.MOTION_BLUR_RAND = False

    # NOTE: add light rendering and keypoint
    _C.LIGHT_RENDER = False
    _C.LIGHT_HIGH = 100
    _C.KEY_POINT = None

    # NOTE: add path motion blur
    _C.VANISHING_POINT = None
    _C.PATH_BLUR_CONS = False
    _C.PATH_BLUR_VAR = False
    
    # NOTE: check if using 2pcnet's augmentation (default is True)
    _C.TWO_PC_AUG = True