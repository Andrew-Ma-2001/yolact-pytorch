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


def save_image_with_image_id(image_path, save_path, image_ids):
    # Check if save_path exists
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # Find all the image name under the image_path
    image_names = os.listdir(image_path)
    # Filter out files that are not images
    image_names = [image_name for image_name in image_names if image_name.endswith('.png')]

    for image_id in tqdm(image_ids):
        # Filter out the image name with the image_id
        image_name = [image_name for image_name in image_names if str(image_id) in image_name]
        # Save the images to the save_path
        for image_name in image_names:
            image = Image.open(os.path.join(image_path, image_name))
            image.save(os.path.join(save_path, image_name))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--config', type=str, default='config/resnet50.yaml', help='Path to the configuration file')
    parser.add_argument('--map_mode', type=int, default=4, help='Map mode for evaluation')
    args = parser.parse_args()

    # Load hyperparameters from yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    map_mode = args.map_mode

    PLOT = False
    PLOT_RESULT = False  
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、计算指标。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅计算指标。
    #   map_mode为3代表仅仅获得预测结果
    #   map_mode为4代表仅仅计算指标一，人像重叠度
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
    save_path = osp.join(map_out_path, 'val_result')
    save_result_path = osp.join(map_out_path, 'all_result')

    if not osp.exists(save_path):
        os.makedirs(save_path)
        # Check if any image is in the save_path, remove them if there are any
        image_ids = os.listdir(save_path)
        image_ids = [int(image_id.split('.')[0]) for image_id in image_ids]
        for image_id in image_ids:
            os.remove(osp.join(save_path, str(image_id) + '.png'))

    
    if not osp.exists(save_result_path):
        os.makedirs(save_result_path)
        # Check if any image is in the save_path, remove them if there are any
        image_ids = os.listdir(save_path)
        image_ids = [int(image_id.split('.')[0]) for image_id in image_ids]
        for image_id in image_ids:
            os.remove(osp.join(save_path, str(image_id) + '.png'))

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
        print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')


        print(f"\nCalculating the person overlap ratio...")
        yolact = YOLACT(confidence = 0.5, nms_iou = 0.3, classes_path = classes_path, model_path = model_path)
        random.seed(0)

        total = 0
        correct = 0

        real_image_count = 0
        correct_image = 0

        for i, id in enumerate(tqdm(ids)):

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
        
            if correct_image + wrong_person > 0:
                real_image_count += 1

            if PLOT_RESULT:
                detect_img = yolact.detect_image(image)

                for annotation in gt_annotation:
                        bbox = annotation['bbox']
                        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='b', linewidth=1))

                plt.imshow(detect_img)
                plt.savefig(f"{save_result_path}/{id}.png")
                # plt.show()
                plt.clf()


        print("The result is:")
        print(correct / total)

        print("Total Eval BBox")
        print(total)

        print("Total Eval Image")
        print(real_image_count)