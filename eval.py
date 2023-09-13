import os
import os.path as osp

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import get_classes, get_coco_label_map
from utils.utils_map import Make_json, prep_metrics
from yolact import YOLACT

import yaml
import argparse
import json

class PersonEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)

    def enlarge_and_evaluate(self, scale=1.2):
        """
        Enlarge the bounding box of the person by a scale around the center point and compare with ground truth if it completely contains the ground truth set acc+1, else set acc+0.
        """
        acc = 0
        total = 0
        for imgId in self.params.imgIds:
            p = self.params
            g = self._gts[imgId, p.catIds[0]]
            d = self._dts[imgId, p.catIds[0]]
            for gt, dt in zip(g, d):
                total += 1
                gt_box = gt['bbox']
                dt_box = dt['bbox']
                dt_box_center = [dt_box[0] + dt_box[2] / 2, dt_box[1] + dt_box[3] / 2]
                dt_box_enlarged = [dt_box_center[0] - dt_box[2] * scale / 2, dt_box_center[1] - dt_box[3] * scale / 2, dt_box[2] * scale, dt_box[3] * scale]
                if dt_box_enlarged[0] <= gt_box[0] and dt_box_enlarged[1] <= gt_box[1] and dt_box_enlarged[0] + dt_box_enlarged[2] >= gt_box[0] + gt_box[2] and dt_box_enlarged[1] + dt_box_enlarged[3] >= gt_box[1] + gt_box[3]:
                    acc += 1
        return acc / total if total > 0 else 0

    def batch_evaluate(self, dir, total_size=1000):
        """
        Evaluate the results by running on total_size of images and save the results to a directory in images and mask drawn with the results.
        """
        from PIL import ImageDraw
        for imgId in self.params.imgIds[:total_size]:
            img_data = self.cocoGt.loadImgs(imgId)[0]
            img = Image.open(os.path.join(dir, img_data['file_name']))
            draw = ImageDraw.Draw(img)
            for ann in self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=imgId)):
                bbox = ann['bbox']
                draw.rectangle([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])], outline='red')
            for ann in self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=imgId)):
                bbox = ann['bbox']
                draw.rectangle([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])], outline='blue')
            img.save(os.path.join(dir, 'results', img_data['file_name']))





if __name__ == '__main__':
    # TODO 三件事情
    # TODO 2. 得到的 人的预测框以中心点放大 1.2 倍，看是否全部囊扩 ground truth，是则为正确，不是则为错误；计算 Accuracy
    # TODO 3. 批量跑 1000 张结果

    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--config', type=str, default='config/resnet50.yaml', help='Path to the configuration file')
    parser.add_argument('--map_mode', type=int, default=0, help='Map mode for evaluation')
    args = parser.parse_args()

    # Load hyperparameters from yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    map_mode = args.map_mode

    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、计算指标。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅计算指标。
    #-------------------------------------------------------------------------------------------------------------------#
    # map_mode        = 2
    #-------------------------------------------------------#
    #   评估自己的数据集必须要修改
    #   所需要区分的类别对应的txt文件
    #-------------------------------------------------------#
    # classes_path    = 'model_data/coco_classes.txt'   
    #-------------------------------------------------------#
    #   获得测试用的图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #-------------------------------------------------------#
    # Image_dir       = "/home/public/datasets/coco/val2017"
    # Json_path       = "/home/public/datasets/coco/annotations/instances_val2017.json"
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #   里面存放了一些json文件，主要是检测结果。
    #-------------------------------------------------------#
    # map_out_path    = 'map_out'
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    classes_path = config['general']['classes_path']
    model_path = config['saving']['save_dir'] + '/best_epoch_weights.pth'
    Image_dir = config['data']['val_image_path']
    Json_path = config['data']['val_annotation_path']
    map_out_path    = config['saving']['save_dir']


    test_coco       = COCO(Json_path)
    class_names, _  = get_classes(classes_path)
    COCO_LABEL_MAP  = get_coco_label_map(test_coco, class_names)
    
    ids         = list(test_coco.imgToAnns.keys())

    #------------------------------------#
    #   创建文件夹
    #------------------------------------#
    if not osp.exists(map_out_path):
        os.makedirs(map_out_path)
        
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        # yolact      = YOLACT(confidence = 0.05, nms_iou = 0.5)
        yolact      = YOLACT(confidence = 0.05, nms_iou = 0.5, classes_path = classes_path, model_path = model_path)
        print("Load model done.")
        
        print("Get predict result.")
        make_json   = Make_json(map_out_path, COCO_LABEL_MAP)
        for i, id in enumerate(tqdm(ids)):
            image_path  = osp.join(Image_dir, test_coco.loadImgs(id)[0]['file_name'])
            image       = Image.open(image_path)

            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = yolact.get_map_out(image)
            if box_thre is None:
                continue
            prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, id, make_json)
        make_json.dump()
        print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')

    if map_mode == 0 or map_mode == 2:
        bbox_dets = test_coco.loadRes(osp.join(map_out_path, "bbox_detections.json"))
        mask_dets = test_coco.loadRes(osp.join(map_out_path, "mask_detections.json"))

        print('\nEvaluating BBoxes:')
        # NOTE: pass catIds parameter here if you want to limit the evaluation to a specific category.
        # bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox')
        print('Tesing Object: Person')
        bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox')
        bbox_eval.params.catIds = [1]
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()
        # person_eval = PersonEval(test_coco, bbox_dets, 'bbox')
        # person_eval.params.catIds = [1]
        # print(person_eval.enlarge_and_evaluate())

        print('\nEvaluating Masks:')
        # bbox_eval = COCOeval(test_coco, mask_dets, 'segm')
        mask_eval = COCOeval(test_coco, mask_dets, 'segm')
        mask_eval.params.catIds = [1]
        mask_eval.evaluate()
        mask_eval.accumulate()
        mask_eval.summarize()


        """
        The results you see are the evaluation metrics for the object detection model you are using, specifically for the "person" class. The metrics are calculated using the COCO evaluation tool. Here's a brief explanation of each metric:

        - Average Precision (AP): This is the average of precisions at different recall values. It's a popular metric in object detection. The AP is calculated at different Intersection over Union (IoU) thresholds, typically from 0.5 to 0.95 (0.5:0.95 in the output). The AP is also calculated for different area sizes of the object, small, medium, and large.
        
        - Average Recall (AR): This is the average of maximum recall given some numbers of detections per image, across all categories. Like AP, AR is also calculated for different area sizes of the object.
        
        In your output:
        
        - The first block of results is for bounding box detection (bbox). The AP for IoU=0.50:0.95 is 0.395, which means the model has 39.5% precision on average across different IoU thresholds from 0.5 to 0.95. The AR for maxDets=100 is 0.476, which means the model has 47.6% recall on average when allowing up to 100 detections per image.
        
        - The second block of results is for instance segmentation (segm). The AP for IoU=0.50:0.95 is 0.318, which means the model has 31.8% precision on average for instance segmentation across different IoU thresholds from 0.5 to 0.95. The AR for maxDets=100 is 0.403, which means the model has 40.3% recall on average when allowing up to 100 detections per image.
        
        These metrics help you understand how well your model is performing. Higher values for these metrics are better.
        """
