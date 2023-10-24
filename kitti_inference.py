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

import torch.onnx
from torch.onnx import OperatorExportTypes

import time

log = src.utils.get_pylogger(__name__)

try: 
    import onnxruntime
    import openvino.runtime as ov
except ImportError:
    log.warning("ONNX and OpenVINO not installed")

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def get_yolo3d_model(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages #TODO: depricated
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
        regressor = core.compile_model(model, 'CPU') #TODO: change to config.get("device")
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
    
        # detect object with Detector
        start_detect = time.time()
        # dets = detector(img)
        try:
            dets = detector(img).crop(save=config.get("save_det2d"))
        except ValueError:
            continue
        avg_time["detector"] += time.time() - start_detect

        # dimension averages
        DIMENSION = []

        # loop thru detections
    
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det["label"].split(" ")[0].capitalize()
            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            box = [box.cpu().numpy() for box in det["box"]]
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = box[0], box[1], box[2], box[3]

            # preprocess img with torch.transforms
            crop = preprocess(cv2.resize(det["im"], (224, 224)))
            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))

            start_reg = time.time()
            # regress 2D bbox with Regressor
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)
                
                # TODO: use this to change to libtorch
                output = regressor(crop)
                orient = output[0]
                conf = output[1]
                dim = output[2]
                
                traced_script_module = torch.jit.trace(regressor, (crop))
                traced_script_module.save("weights/yolo_libtorch_model_3d.pth")
                
                onnx_model_save_path = "weights/yolo_onnx_model_3d.onnx"
                onnx_fp16_model_save_path = "weights/yolo_onnx_model_3d_fp16.onnx"
                # dynamic_axes = {"image": {0: "batch"}, 
                #                 "orient": {0: "batch", 1: str(2), 2: str(2)}, # for fp32 onnx model
                #                 "conf": {0: "batch"}, 
                #                 "dim": {0: "batch"}}
                if True:
                    torch.onnx.export(regressor, crop, onnx_model_save_path, opset_version=11,
                                verbose=False, export_params=True, #operator_export_type=OperatorExportTypes.ONNX,
                                input_names=['image'], output_names=['orient','conf','dim']
                                #,dynamic_axes=dynamic_axes
                                )
                    print("Please check onnx model in ", onnx_model_save_path)
                # conda install -c conda-forge onnx
                
                import onnx
                onnx_model = onnx.load(onnx_model_save_path)
                
                # for dla&trt speedup
                from onnxmltools.utils import float16_converter
                trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
                onnx.save_model(trans_model, onnx_fp16_model_save_path)

                exit()

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
        cv2.imwrite('./eval_kitti/crop/img.jpg', img)

        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./eval_kitti/crop/demo.jpg', crop)

    except cv2.error:
        print("pt1 is ", pt1, " pt2 is ", pt2)
        print("image shape is ", img.shape)
        print("box_2d is ", box_2d)

    # apply transform for batch
    batch = process(crop)

    return batch

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def kitti_inference_label(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages #TODO: depricated
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
        regressor = core.compile_model(model, 'CPU') #TODO: change to config.get("device")
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

    # loop thru images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        # read gt image ./eval_kitti/image_2_val/
        img_id = img_path[-10:-4]
        
        # dt result
        result_label_root_path = './eval_kitti/result/'
        f = open(result_label_root_path + img_id + '.txt', 'w')
        
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_label_root_path = './eval_kitti/label_2_val/'
        gt_f = gt_label_root_path + img_id + '.txt'

        dets = []
        with open(gt_f, 'r') as file:
                content = file.readlines()
                for i in range(len(content)):
                    gt = content[i].split()
                    top_left, bottom_right = (int(float(gt[4])), int(float(gt[5]))), (int(float(gt[6])), int(float(gt[7])))
                    
                    bbox_2d = [top_left, bottom_right]
                    label = gt[0]

                    dets.append(Bbox(bbox_2d, label))
                     
        # dimension averages 
        DIMENSION = []

        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det.detected_class
            if(obj.name == 'DontCare'):
                continue
            if(obj.name == 'Misc'):
                continue
            if(obj.name == 'Person_sitting'):
                continue

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
            
            obj.alpha = recover_angle(orient, conf, 2)
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            obj.score = 1.0
            output_line = obj.member_to_list()
            output_line = " ".join([str(i) for i in output_line])
            
            f.write(output_line + '\n')
        f.close()

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def kitti_inference_image(config: DictConfig):
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
        if img.shape != (375, 1242, 3):
            # crop center of image to 1242 x 375
            src_h, src_w, _ = img.shape
            dst_h, dst_w = 375, 1242
            dif_h, dif_w = src_h - dst_h, src_w - dst_w
            img = img[dif_h // 2 : src_h - dif_h // 2, dif_w // 2 : src_w - dif_w // 2, :]

        img_id = img_path[-10:-4]
        
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_label_root_path = './eval_kitti/label_2/'
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
            if(obj.name == 'DontCare'):
                continue
            if(obj.name == 'Misc'):
                continue
            if(obj.name == 'Person_sitting'):
                continue
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
            # dim[0], dim[1], dim[2] = det.h, det.w, det.l
            
            obj.alpha = recover_angle(orient, conf, 2)
            # obj.alpha = det.alpha
    
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            # obj.rot_global = det.ry
            
            # TODO:check
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
def inference_stage2(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages #TODO: depricated
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
        regressor = core.compile_model(model, 'CPU') #TODO: change to config.get("device")
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

    # loop thru images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        # read gt 
        img_id = img_path[-10:-4]

        result_label_root_path = './eval/result/'
        f = open(result_label_root_path + img_id + '.txt', 'w')
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dets = []
        gt_label_root_path = './eval/label_2_val/'
        gt_f = gt_label_root_path + img_id + '.txt'
        
        dets = detector(img).crop(save=config.get("save_det2d"))

        # dimension averages #TODO: depricated
        DIMENSION = []

        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = class_to_labels(int(det["label"].split(" ")[0].capitalize()))

            
            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            box = [box.cpu().numpy() for box in det["box"]]
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = box[0], box[1], box[2], box[3]

            # preprocess img with torch.transforms
            crop = preprocess(cv2.resize(det["im"], (224, 224)))
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
                dim += class_averages.get_item(class_to_labels(det["cls"].cpu().numpy()))
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]
            
            obj.alpha = recover_angle(orient, conf, 2) # 8
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            obj.score = det["conf"].numpy()
            output_line = obj.member_to_list()
            output_line = " ".join([str(i) for i in output_line])
            
            f.write(output_line + '\n')
        f.close()


def detector_yolov5(model_path: str, cfg_path: str, classes: int, device: str):
    """YOLOv5 detector model"""
    sys.path.append(str(root / "yolov5"))

    # NOTE: ignore import error
    from models.common import AutoShape
    from models.yolo import Model
    from utils.general import intersect_dicts
    from utils.torch_utils import select_device

    device = select_device(
        ("0" if torch.cuda.is_available() else "cpu") if device is None else device
    )

    model = Model(cfg_path, ch=3, nc=classes)
    ckpt = torch.load(model_path, map_location=device)  # load
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if len(ckpt["model"].names) == classes:
        model.names = ckpt["model"].names  # set class names attribute
    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS

    return model.to(device)

def class_to_labels(class_: int, list_labels: List = None):

    if list_labels is None:
        # TODO: change some labels mistakes
        # list_labels = ['car', 'van', 'truck', 'pedestrian', 'cyclist']
        list_labels = ['Car', 'Cyclist', 'Truck','Van', 'Pedestrian', 'Tram']

    return list_labels[int(class_)]

def copy_eval_label():
    label_path = './eval/ImageSets/val.txt'
    label_root_path = './eval/label_2/'
    label_save_path = './eval/label_2_val/'

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
    label_path = './eval/ImageSets/val.txt'
    img_root_path = './eval/image_2/'
    img_save_path = './eval/image_2_val'

    # get all labels
    label_files = []
    sum_number = 0
    with open(label_path, 'r') as file:
        img_id = file.readlines()
        for id in img_id:
            img_path = img_root_path + id[:6] + '.png'
            img = cv2.imread(img_path)
            cv2.imwrite(f'{img_save_path}/{id[:6]}.png', img)


if __name__ == "__main__":

    # copy_eval_label()
    # copy_eval_image()
    
    # generate libtorch model
    get_yolo3d_model()

    # inference_stage2()
    
    # kitti_inference_image: inference for kitti bev and 3d image, without model
    # kitti_inference_image()
    
    # kitti_inference_label: for kitti gt label, predict without model
    # kitti_inference_label()
