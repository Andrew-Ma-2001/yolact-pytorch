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
import numpy as np

import matplotlib.pyplot as plt
import cv2


import random


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


def cover_iou(box1, box2, scale=1.2):
    """
    Enlarge box1 by a scale and check if it fully covers box2.
    """
    # Enlarge box1
    box1_center = [box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
    box1_enlarged = [box1_center[0] - box1[2] * scale / 2, box1_center[1] - box1[3] * scale / 2, box1[2] * scale, box1[3] * scale]

    # Check if box1_enlarged fully covers box2
    if box1_enlarged[0] <= box2[0] and box1_enlarged[1] <= box2[1] and box1_enlarged[0] + box1_enlarged[2] >= box2[0] + box2[2] and box1_enlarged[1] + box1_enlarged[3] >= box2[1] + box2[3]:
        return True
    else:
        return False
    

def bbox_iou_coco(box1, box2):
    """
    This function calculates intersection over union (IoU) given two bounding boxes in COCO format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Determine the (x, y)-coordinates of the intersection rectangle
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = min(x1 + w1, x2 + w2) - x_inter
    h_inter = min(y1 + h1, y2 + h2) - y_inter

    if w_inter < 0 or h_inter < 0:  # if there is no overlap
        return 0.0

    # Compute the area of intersection rectangle
    inter_area = w_inter * h_inter

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the intersection over union by taking the intersection area and dividing it 
    # by the sum of prediction + ground-truth areas - the intersection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    # return the intersection over union value
    return iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--config', type=str, default='config/resnet50.yaml', help='Path to the configuration file')
    parser.add_argument('--map_mode', type=int, default=4, help='Map mode for evaluation')
    args = parser.parse_args()

    # Load hyperparameters from yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    map_mode = args.map_mode
    PLOT = True
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
    model_path = 'logs/resnet50/best_epoch_weights.pth'
    Image_dir = config['data']['val_image_path']
    Json_path = 'model_data/instances_val2017_person_5.json'
    map_out_path    = config['saving']['save_dir']
    save_path = osp.join(map_out_path, 'val_result')

    if not osp.exists(save_path):
        os.makedirs(save_path)

    test_coco       = COCO(Json_path)
    class_names, _  = get_classes(classes_path)
    COCO_LABEL_MAP  = get_coco_label_map(test_coco, class_names)
    
    ids         = list(test_coco.imgToAnns.keys())

    #------------------------------------#
    #   创建文件夹
    #------------------------------------#
    if not osp.exists(map_out_path):
        os.makedirs(map_out_path)
        

    if map_mode == 3:
        ids = ids[:200]
        print("Initializing YOLACT for detection...")
        yolact      = YOLACT(confidence = 0.6, nms_iou = 0.6, classes_path = classes_path, model_path = model_path)
        print("YOLACT initialized.")
        for i, id in enumerate(tqdm(ids)):
            image_path  = osp.join(Image_dir, test_coco.loadImgs(id)[0]['file_name'])
            # print(f"Processing image: {image_path}")
            image       = Image.open(image_path)
            r_image = yolact.detect_image(image)
            # print("Image processed.")
            # Set up an index and save to the mapout path
            r_image.save(osp.join(save_path, str(i) + '.jpg'))



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


    if map_mode == 4:
        yolact = YOLACT(confidence = 0.5, nms_iou = 0.3, classes_path = classes_path, model_path = model_path)

        total = 0
        correct = 0

        import os

        # Get all files in the current directory
        files = os.listdir('/home/mayanze/PycharmProjects/yolact-pytorch-main/logs/resnet50/val_result')

        # Filter out the .png files
        png_files = [f for f in files if f.endswith('.png')]

        # Remove the .png extension and convert to int
        image_numbers = [int(f.replace('.png', '')) for f in png_files]

        # Use Random to remove 50% of the image numbers
        image_numbers = random.sample(image_numbers, int(len(image_numbers) * 0.5))


        image_ids = ['8844', '9769', '17379', '35062', '48504', '58393', '59598', '66706', '74733', '76416', '97337', '98716', '110042', '111086',
             '131556', '153527', '172935', '196442', '236599', '274272', '305309', '306136', '309713', '315492', '336356', '345361',
             '355817', '359677', '369323', '369541', '381587', '385190', '391722', '395575', '397681', '407943', '423123', '424135',
             '425221', '437514', '455085', '456303', '458755', '461751', '477805', '480275', '493286', '514586', '530854', '536038',
             '538236', '541773', '554266', '557672', '568584'
             ]
        real_image_count = 0
        correct_image = 0

        for i, id in enumerate(tqdm(ids)):

            # if str(id) in image_ids:
            #     continue

            if id in image_numbers:
                continue

            image_path  = osp.join(Image_dir, test_coco.loadImgs(id)[0]['file_name'])
            # Load in the ground true annotations
            gt_annotation = test_coco.loadAnns(test_coco.getAnnIds(imgIds=id, iscrowd=None))

            image       = Image.open(image_path)
            result = yolact.get_single_image(image)

            longest_side = max(image.size)

            # Use IOU to find the best match
            # Compute the IOU between each result bbox and each gt bbox
            # If the IOU is larger than 0.5, then it is a match
            # If there are multiple matches, choose the one with the largest IOU
            # If there are no matches, then it is a false positive
            total_person = len([annotation for annotation in gt_annotation if annotation['category_id'] == 1])
            wrong_person = 0
            correct_image = 0
            
            for annotation in gt_annotation:
                result_bbox = result['boxes']
                bbox = annotation['bbox']

                # Skip the bbox if its height is less than 5% of the longest side
                (x,y,w,h) = bbox

                if h < longest_side * 0.05 or w < longest_side * 0.05:
                    continue
                
                # Skip the bbox if any point of the bbox is in the 5% edge of the image
                # if x < image.size[0] * 0.05 or y < image.size[1] * 0.05:
                #     continue

                # if x + w > image.size[0] * 0.95 or y + h > image.size[1] * 0.95:
                #     continue

                if total_person > 5:
                    continue

                if result_bbox is None:
                    total += 1
                    continue

                total += 1

                # Change bbox format in result_bbox from (x1, y1, x2, y2) to (x, y, w, h)
                result_bbox = [[rb[0], rb[1], rb[2] - rb[0], rb[3] - rb[1]] for rb in result_bbox]

                ious = np.array([bbox_iou_coco(bbox, rb) for rb in result_bbox])
                best_matches = np.where(ious > 0.5)[0]
                if len(best_matches) > 0:
                    best_match = best_matches[np.argmax(ious[best_matches])]
                    # if cover_iou(result_bbox[best_match], bbox):
                        # correct += 1
                    correct += 1
                    correct_image += 1
                else:
                    wrong_person += 1
                    # print('No match found for ground truth bbox', bbox)


            if wrong_person > 0:
                detect_img = yolact.detect_image(image)

                if PLOT:
                    for annotation in gt_annotation:
                        bbox = annotation['bbox']

                        # Filter out the image which smallest side is less than 5% of the image longest side
                        if min(bbox[2], bbox[3]) < 0.05 * max(image.size):
                            continue

                        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='b', linewidth=1))

                    plt.imshow(detect_img)
                    plt.savefig(f"{save_path}/{id}.png")
                
                    # plt.show()
                    plt.clf()
        
            if correct_image > 0:
                real_image_count += 1

        print("The result is:")
        print(correct / total)

        print("Total Eval BBox")
        print(total)

        print("Total Eval Image")
        print(real_image_count)

        """
        The results you see are the evaluation metrics for the object detection model you are using, specifically for the "person" class. The metrics are calculated using the COCO evaluation tool. Here's a brief explanation of each metric:

        - Average Precision (AP): This is the average of precisions at different recall values. It's a popular metric in object detection. The AP is calculated at different Intersection over Union (IoU) thresholds, typically from 0.5 to 0.95 (0.5:0.95 in the output). The AP is also calculated for different area sizes of the object, small, medium, and large.
        
        - Average Recall (AR): This is the average of maximum recall given some numbers of detections per image, across all categories. Like AP, AR is also calculated for different area sizes of the object.
        
        In your output:
        
        - The first block of results is for bounding box detection (bbox). The AP for IoU=0.50:0.95 is 0.395, which means the model has 39.5% precision on average across different IoU thresholds from 0.5 to 0.95. The AR for maxDets=100 is 0.476, which means the model has 47.6% recall on average when allowing up to 100 detections per image.
        
        - The second block of results is for instance segmentation (segm). The AP for IoU=0.50:0.95 is 0.318, which means the model has 31.8% precision on average for instance segmentation across different IoU thresholds from 0.5 to 0.95. The AR for maxDets=100 is 0.403, which means the model has 40.3% recall on average when allowing up to 100 detections per image.
        
        These metrics help you understand how well your model is performing. Higher values for these metrics are better.
        """
