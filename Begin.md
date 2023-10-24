conda create -n Yolo3D python=3.8 numpy

conda activate Yolo3D

pip install -r requirements.txt

python src/train.py \
  experiment=sample

python inference.py \
  source_dir=./eval/image_2 \
  detector.model_path=./weights/backup/detector_yolov5s.pt \
  regressor_weights=./weights/pytorch-test.pt

<!-- 更改 assets/global_calib.txt 内参矩阵 -->
python evaluate.py \
  detector.model_path=./weights/detector_yolov5s.pt \
  regressor_weights=./weights/regressor_resnet18.pt

<!-- inference: trans .ckpt to torch .pt model, modify configs/convert.yaml -->
python convert.py

<!-- then trans pytorch-test.pt to libtorch model -->
1. 修改模型返回结果 src/models/regressor.py
python My_inference.py \
  source_dir=./data/KITTI/test \
  detector.model_path=./weights/backup/detector_yolov5s.pt \
  regressor_weights=./weights/backup/regressor_resnet18.pt

<!-- evaluate result -->
1. 生成class_averages标签
2. 修改class_to_labels中list_labels映射
3. inference中执行kitti_inference_label()
4. 指定图片所在地址source_dir
5. 指定dt地址result_label_root_path
6. 指定gt地址gt_label_root_path
python kitti_inference.py \
  source_dir=./eval_kitti/image_2_val \
  detector.model_path=./weights/backup/kitti_yolov5s.pt \
  detector.cfg_path=./yolov5/models/yolov5s-kitti.yaml \
  detector.classes=6 \
  regressor_weights=./weights/backup/regressor_resnet18.pt


```json
  // kitti_inference_label
  "args": [
      "source_dir=./eval_kitti/image_2_val",
      "detector.model_path=./weights/backup/kitti_yolov5s.pt",
      "detector.cfg_path=./yolov5/models/yolov5s-kitti.yaml",
      "detector.classes=6",
      "regressor_weights=weights/backup/regressor_resnet18.pt"
  ],
  // kitti_inference_image
  "args": [
      "source_dir=./eval_kitti/image_2",
      "detector.model_path=./weights/backup/kitti_yolov5s.pt",
      "detector.cfg_path=./yolov5/models/yolov5s-kitti.yaml",
      "detector.classes=6",
      "regressor_weights=weights/backup/regressor_resnet18.pt"
  ],
  // L4-inference_image&label
  "args": [
      "source_dir=./eval_L4/image_2",
      "detector.model_path=./weights/backup/detector_yolov5s.pt",
      "regressor_weights=./weights/pytorch-test.pt"
  ],
```

# train yolov5
python train.py --data kitti.yaml --weights '' --cfg yolov5s-kitti.yaml --img 640


-------------------------TODO------------------------
python torch2onnx.py \
    --reg_weights weights/back/resnet18.pkl \
    --model_select resnet18  \
    --output_path runs/models/ 

TODO:
1. use tensorboard to show loss details
2. update requirements

case1:
AttributeError: ‘Upsample‘ object has no attribute ‘recompute_scale_factor‘
[https://blog.csdn.net/Thebest_jack/article/details/124723687]

case2:
cv2.error: Caught error in DataLoader worker process 1
[https://github.com/ultralytics/yolov3/issues/1721]