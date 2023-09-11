import os
import os.path as osp

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import get_classes, get_coco_label_map
from utils.utils_map import Make_json, prep_metrics
from yolact import YOLACT



def enlarge_bbox(box, scale=1.2):
    """
    Enlarge a bounding box by a scale around the center point
    """
    x, y, w, h = box
    center_x, center_y = x + w / 2, y + h / 2
    new_w, new_h = w * scale, h * scale
    new_x, new_y = center_x - new_w / 2, center_y - new_h / 2
    return [new_x, new_y, new_w, new_h]

def is_bbox_inside(bbox1, bbox2):
    """
    Check if bbox1 is inside bbox2
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

def prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, id, make_json):
    # ... original code ...

    # Enlarge the bounding boxes and calculate accuracy
    correct_count = 0
    total_count = len(box_thre)
    for i in range(total_count):
        if class_ids[i] == 1:  # "person" class
            enlarged_box = enlarge_bbox(box_thre[i])
            for gt_box in test_coco.imgToAnns[id]:
                if is_bbox_inside(enlarged_box, gt_box['bbox']):
                    correct_count += 1
                    break
    accuracy = correct_count / total_count

    # ... original code ...





if __name__ == '__main__':
    # TODO 三件事情
    # TODO 2. 得到的 人的预测框以中心点放大 1.2 倍，看是否全部囊扩 ground truth，是则为正确，不是则为错误；计算 Accuracy
    # TODO 3. 批量跑 1000 张结果


    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、计算指标。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅计算指标。
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 2
    #-------------------------------------------------------#
    #   评估自己的数据集必须要修改
    #   所需要区分的类别对应的txt文件
    #-------------------------------------------------------#
    classes_path    = 'model_data/coco_classes.txt'   
    #-------------------------------------------------------#
    #   获得测试用的图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #-------------------------------------------------------#
    Image_dir       = "/home/public/datasets/coco/val2017"
    Json_path       = "/home/public/datasets/coco/annotations/instances_val2017.json"
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #   里面存放了一些json文件，主要是检测结果。
    #-------------------------------------------------------#
    map_out_path    = 'map_out'
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
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
        yolact      = YOLACT(confidence = 0.05, nms_iou = 0.5)
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
        bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox', catIds=[1])
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nEvaluating Masks:')
        # bbox_eval = COCOeval(test_coco, mask_dets, 'segm')
        mask_eval = COCOeval(test_coco, mask_dets, 'segm', catIds=[1])
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

