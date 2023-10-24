""" Inference Code """

from typing import List
from PIL import Image
import cv2
from glob import glob
import numpy as np

import torch
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule
from src.utils import Calib
from src.utils.averages import ClassAverages
from src.utils.Plotting import calc_alpha, plot_3d_box
from src.utils.Math import calc_location, compute_orientaion, recover_angle, translation_constraints
from src.utils.Plotting import calc_theta_ray
from src.utils.Plotting import Plot3DBoxBev

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys
import pyrootutils
import src.utils
from src.utils.utils import KITTIObject

import time

log = src.utils.get_pylogger(__name__)

try: 
    import onnxruntime
    import openvino.runtime as ov
except ImportError:
    log.warning("ONNX and OpenVINO not installed")

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

class Bbox:
    def __init__(self, box_2d, label, h, w, l, tx, ty, tz, ry, alpha):
        self.box_2d = box_2d
        self.detected_class = label
        self.w = w
        self.h = h
        self.l = l
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.ry = ry
        self.alpha = alpha
        
        
def format_img(img, box_2d):
    # transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    process = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # crop image
    pt1, pt2 = box_2d[0], box_2d[1]

    point_list1 = [pt1[0], pt1[1]]
    point_list2 = [pt2[0], pt2[1]]
    
    if point_list1[0] < 0:
        point_list1[0] = 0
    if point_list1[1] < 0:
        point_list1[1] = 0
    if point_list2[0] < 0:
        point_list2[0] = 0
    if point_list2[1] < 0:
        point_list2[1] = 0
        
    if point_list1[0] >= img.shape[1]:
        point_list1[0] = img.shape[1] - 1
    if point_list2[0] >= img.shape[1]:
        point_list2[0] = img.shape[1] - 1
    if point_list1[1] >= img.shape[0]:
        point_list1[1] = img.shape[0] - 1
    if point_list2[1] >= img.shape[0]:
        point_list2[1] = img.shape[0] - 1
        
    crop = img[point_list1[1]:point_list2[1]+1, point_list1[0]:point_list2[0]+1]
    
    try: 
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    except cv2.error:
        print("pt1 is ", pt1, " pt2 is ", pt2)
        print("image shape is ", img.shape)
        print("box_2d is ", box_2d)

    # apply transform for batch
    batch = process(crop)

    return batch


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference_image(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages
    class_averages = ClassAverages()

    # time
    avg_time = {
        "initiate_detector": 0,
        "initiate_regressor": 0,
        "detector": 0,
        "regressor": 0,
        "plotting": 0,
    }

    # initialize detector model
    start_detector = time.time()
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)
    avg_time["initiate_detector"] = time.time() - start_detector

    # initialize regressor model
    start_regressor = time.time()
    if config.get("inference_type") == "pytorch":
        # pytorch regressor model
        log.info(f"Instantiating regressor <{config.model._target_}>")
        regressor: LightningModule = hydra.utils.instantiate(config.model)
        regressor.load_state_dict(torch.load(config.get("regressor_weights"), map_location="cpu"))
        regressor.eval().to(config.get("device"))
    elif config.get("inference_type") == "onnx":
        # onnx regressor model
        log.info(f"Instantiating ONNX regressor <{config.get('regressor_weights').split('/')[-1]}>")
        regressor = onnxruntime.InferenceSession(config.get("regressor_weights"), providers=providers)
        input_name = regressor.get_inputs()[0].name
    elif config.get("inference_type") == "openvino":
        # openvino regressor model
        log.info(f"Instantiating OpenVINO regressor <{config.get('regressor_weights').split('/')[-1]}>")
        core = ov.Core()
        model = core.read_model(config.get("regressor_weights"))
        regressor = core.compile_model(model, 'CPU')
        infer_req = regressor.create_infer_request()
    avg_time["initiate_regressor"] = time.time() - start_regressor

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("output_dir"), exist_ok=True)

    # TODO: inference on video
    # loop thru images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        # Initialize object and plotting modules
        plot3dbev = Plot3DBoxBev(P2)

        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # check if image shape 1242 x 375
        if img.shape != (1080, 1920, 3):
            # crop center of image to 1242 x 375
            src_h, src_w, _ = img.shape
            dst_h, dst_w = 1080, 1920
            dif_h, dif_w = src_h - dst_h, src_w - dst_w
            img = img[dif_h // 2 : src_h - dif_h // 2, dif_w // 2 : src_w - dif_w // 2, :]

        img_id = img_path[-10:-4]
        
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_label_root_path = './eval_L4/label_2/'
        gt_label_root_path = './data/KITTI/label_2/'
        
        # gt_label_root_path = './eval_L4/test_for_3dcalculate_label/'
        gt_f = gt_label_root_path + img_id + '.txt'

        dets = []
        try:
            with open(gt_f, 'r') as file:
                content = file.readlines()
                for i in range(len(content)):
                    gt = content[i].split()
                    top_left, bottom_right = (int(float(gt[4])), int(float(gt[5]))), (int(float(gt[6])), int(float(gt[7])))
                    
                    bbox_2d = [top_left, bottom_right]
                    label = gt[0]

                    dets.append(Bbox(bbox_2d, label, float(gt[8]), float(gt[9]), float(gt[10]), float(gt[11]), float(gt[12]), float(gt[13]), float(gt[14]), float(gt[3])))
        except:
            continue
        # dimension averages #TODO: depricated
        DIMENSION = []

        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det.detected_class

            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = det.box_2d[0][0], det.box_2d[0][1], det.box_2d[1][0], det.box_2d[1][1]

            crop = format_img(img, det.box_2d)

            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))

            start_reg = time.time()
            # regress 2D bbox with Regressor
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)
                orient = orient.cpu().detach().numpy()[0, :, :]
                conf = conf.cpu().detach().numpy()[0, :]
                dim = dim.cpu().detach().numpy()[0, :]

            # dimension averages # TODO: depricated
            try:
                dim += class_averages.get_item(obj.name)
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]
                
            dim[0], dim[1], dim[2] = det.h, det.w, det.l
            obj.alpha = recover_angle(orient, conf, 2)
            # obj.alpha = det.alpha
    
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.rot_global = det.ry
            
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            output_line = obj.member_to_list()
            output_line.append(1.0)
            output_line = " ".join([str(i) for i in output_line]) + "\n"

            avg_time["regressor"] += time.time() - start_reg

            # write results
            if config.get("save_txt"):
                with open(f"{config.get('output_dir')}/{img_name}.txt", "a") as f:
                    f.write(output_line)


            if config.get("save_result"):
                start_plot = time.time()
                # dt
                plot3dbev.plot(
                    img=img,
                    class_object=obj.name.lower(),
                    bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                    dim=[obj.h, obj.w, obj.l],
                    loc=[obj.tx, obj.ty, obj.tz],
                    rot_y=obj.rot_global,
                    gt=False
                )
                avg_time["plotting"] += time.time() - start_plot
                # gt
                plot3dbev.plot(
                    img=img,
                    class_object=obj.name.lower(),
                    bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                    dim=[det.h, det.w, det.l],
                    loc=[det.tx, det.ty, det.tz],
                    rot_y=det.ry,
                    gt=True
                )
        # save images
        if config.get("save_result"):
            # cv2.imwrite(f'{config.get("output_dir")}/{img_name}.png', img_draw)
            plot3dbev.save_plot(config.get("output_dir"), img_name)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference_label(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages
    class_averages = ClassAverages()

    # time
    avg_time = {
        "initiate_detector": 0,
        "initiate_regressor": 0,
        "detector": 0,
        "regressor": 0,
        "plotting": 0,
    }

    # initialize detector model
    start_detector = time.time()
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)
    avg_time["initiate_detector"] = time.time() - start_detector

    # initialize regressor model
    start_regressor = time.time()
    if config.get("inference_type") == "pytorch":
        # pytorch regressor model
        log.info(f"Instantiating regressor <{config.model._target_}>")
        regressor: LightningModule = hydra.utils.instantiate(config.model)
        regressor.load_state_dict(torch.load(config.get("regressor_weights"), map_location="cpu"))
        regressor.eval().to(config.get("device"))
    elif config.get("inference_type") == "onnx":
        # onnx regressor model
        log.info(f"Instantiating ONNX regressor <{config.get('regressor_weights').split('/')[-1]}>")
        regressor = onnxruntime.InferenceSession(config.get("regressor_weights"), providers=providers)
        input_name = regressor.get_inputs()[0].name
    elif config.get("inference_type") == "openvino":
        # openvino regressor model
        log.info(f"Instantiating OpenVINO regressor <{config.get('regressor_weights').split('/')[-1]}>")
        core = ov.Core()
        model = core.read_model(config.get("regressor_weights"))
        regressor = core.compile_model(model, 'CPU')
        infer_req = regressor.create_infer_request()
    avg_time["initiate_regressor"] = time.time() - start_regressor

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("output_dir"), exist_ok=True)


    # loop truth images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        # read gt image ./eval_kitti/image_2_val/
        img_id = img_path[-10:-4]
        
        # dt result
        result_label_root_path = './eval_L4/result/'
        f = open(result_label_root_path + img_id + '.txt', 'w')
        
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # gt result
        gt_label_root_path = './eval_L4/label_2/'
        gt_f = gt_label_root_path + img_id + '.txt'

        dets = []
        with open(gt_f, 'r') as file:
                content = file.readlines()
                for i in range(len(content)):
                    gt = content[i].split()
                    top_left, bottom_right = (int(float(gt[4])), int(float(gt[5]))), (int(float(gt[6])), int(float(gt[7])))
                    
                    bbox_2d = [top_left, bottom_right]
                    label = gt[0]

                    dets.append(Bbox(bbox_2d, label, float(gt[8]), float(gt[9]), float(gt[10]), float(gt[11]), float(gt[12]), float(gt[13]), float(gt[14]), float(gt[3])))

        # dimension averages 
        DIMENSION = []

        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det.detected_class

            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = det.box_2d[0][0], det.box_2d[0][1], det.box_2d[1][0], det.box_2d[1][1]

            crop = format_img(img, det.box_2d)

            # # preprocess img with torch.transforms
            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))

            # regress 2D bbox with Regressor
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)

                orient = orient.cpu().detach().numpy()[0, :, :]
                conf = conf.cpu().detach().numpy()[0, :]
                dim = dim.cpu().detach().numpy()[0, :]

            # dimension averages
            try:
                dim += class_averages.get_item(obj.name)
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]
                
            dim[0], dim[1], dim[2] = det.h, det.w, det.l
            obj.alpha = recover_angle(orient, conf, 2)
            # obj.alpha = det.alpha
    
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            # obj.rot_global = det.ry
            
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            obj.score = 1.0
            output_line = obj.member_to_list()
            output_line = " ".join([str(i) for i in output_line])
            
            f.write(output_line + '\n')
        f.close()
        
def copy_eval_label():
    label_path = './eval_L4/ImageSets/val.txt'
    label_root_path = './data/KITTI/label_2/'
    label_save_path = './eval_L4/label_2/'

    # get all labels
    label_files = []
    sum_number = 0
    from shutil import copyfile

    with open(label_path, 'r') as file:
        img_id = file.readlines()
        for id in img_id:
            label_path = label_root_path + id[:6] + '.txt'
            copyfile(label_path, label_save_path + id[:6] + '.txt')

def copy_eval_image():
    label_path = './eval_L4/ImageSets/val.txt'
    img_root_path = './data/KITTI/image_2/'
    img_save_path = './eval_L4/image_2'

    # get all labels
    label_files = []
    sum_number = 0
    with open(label_path, 'r') as file:
        img_id = file.readlines()
        for id in img_id:
            img_path = img_root_path + id[:6] + '.jpg'
            img = cv2.imread(img_path)
            cv2.imwrite(f'{img_save_path}/{id[:6]}.jpg', img)


if __name__ == "__main__":
    # copy_eval_image()
    # copy_eval_label()

    inference_image()

    # inference_label()
