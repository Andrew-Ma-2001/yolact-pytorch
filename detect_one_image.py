from yolact import YOLACT


def run_one_image(model_path, class_path, img):
    """
        This function takes a PIL image as input, performs object detection on it,
        and returns the detected objects, their bounding boxes, classes, and masks.

        Args:
            img (PIL.Image): The input image.

        Returns:
            dict: A dictionary containing the following fields:
                'boxes': A list of bounding boxes of detected objects. Each bounding box is represented by a list of four integers.
                'classes': A list of class IDs of detected objects.
                'scores': A list of confidence scores of detected objects.
                'masks': A list of binary masks of detected objects. Each mask is a 2D numpy array.
    """
    net = YOLACT(model_path = model_path, class_path = class_path)
    result = net.get_single_image(img)
    return result

