import json
import numpy as np
import random

train_image_path        = "/home/public/datasets/coco/train2017"
train_annotation_path   = "/home/public/datasets/coco/annotations/instances_train2017.json"
val_image_path          = "/home/public/datasets/coco/val2017"
val_annotation_path     = "/home/public/datasets/coco/annotations/instances_val2017.json"


def filter_annotations(annotation_path, output_path):
    import time
    start_time = time.time()
    print("Starting to filter annotations...")

    # Load the original COCO annotation file
    load_start_time = time.time()
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    load_end_time = time.time()
    print("Time taken to load the original COCO annotation file: {} seconds".format(load_end_time - load_start_time))

    # Filter the categories and keep only the 'person' category
    filter_cat_start_time = time.time()
    data['categories'] = [category for category in data['categories'] if category['name'] == 'person']
    filter_cat_end_time = time.time()
    print("Time taken to filter the categories: {} seconds".format(filter_cat_end_time - filter_cat_start_time))

    # Get the ID of the 'person' category
    get_id_start_time = time.time()
    person_id = data['categories'][0]['id']
    get_id_end_time = time.time()
    print("Time taken to get the ID of the 'person' category: {} seconds".format(get_id_end_time - get_id_start_time))

    # Filter the annotations and keep only those belong to 'person'
    filter_ann_start_time = time.time()
    data['annotations'] = [annotation for annotation in data['annotations'] if annotation['category_id'] == person_id]
    filter_ann_end_time = time.time()
    print("Time taken to filter the annotations: {} seconds".format(filter_ann_end_time - filter_ann_start_time))

    # Save the filtered data to a new annotation file
    save_start_time = time.time()
    with open(output_path, 'w') as f:
        json.dump(data, f)
        
    save_end_time = time.time()
    print("Time taken to save the filtered data to a new annotation file: {} seconds".format(save_end_time - save_start_time))


    end_time = time.time()
    print("Finished filtering annotations. Time taken: {} seconds".format(end_time - start_time))


def filter_val_images(annotation_path, output_path):
    # Filter the unwanted images
    import os
    # Get all files in the current directory
    files = os.listdir('/home/mayanze/PycharmProjects/yolact-pytorch-main/logs/resnet50/val_result')
    # Filter out the .png files
    png_files = [f for f in files if f.endswith('.png')]
    # Remove the .png extension and convert to int
    image_numbers = [int(f.replace('.png', '')) for f in png_files]
    # Use Random to remove 50% of the image numbers
    image_numbers = random.sample(image_numbers, int(len(image_numbers) * 0.75))
    # Read the coco annotations, here it should be all having one class person
    # Then for every image read the bboxs 
    # Filter out the image annotations if containing more than five persons i.e. meaning that images annotations which have more than 5 should be filtered out
    # Save the filtered annotations to a new file
    import json
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    # Find all the unique image id
    # For each image id, find the total number of bbox annotations
    # If the total number of bbox annotations is more than 5, then remove the annotation with the image id
    # Save the filtered annotations to a new file
    image_ids = [annotation['image_id'] for annotation in data['annotations']]
    unique_image_ids = list(set(image_ids))

    filtered_annotations = []
    for image_id in unique_image_ids:
        image_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
        
        if len(image_annotations) > 5:
            data['annotations'] = [annotation for annotation in data['annotations'] if annotation['image_id'] != image_id]
            continue

        for annotation in image_annotations:
            bbox = annotation['bbox']
            (x, y, w, h) = bbox

            # Remove image annotations where the bounding box is located within the 5% border of the image
            for images in data['images']:
                if images['id'] == image_id:
                    image = images
                    break
            image_width = image['width']
            image_height = image['height']

            longest_side = max(image_width, image_height)
            # Remove image annotations where the bounding box's height or width is less than 5% of the longest side
            if h < longest_side * 0.05 or w < longest_side * 0.05:
                # data['annotations'].remove(annotation)
                continue

            if x < image_width * 0.05 or x + w > image_width * 0.95 or y < image_height * 0.05 or y + h > image_height * 0.95:
                # data['annotations'].remove(annotation)
                continue

            filtered_annotations.append(annotation)

    data['annotations'] = filtered_annotations

    image_ids = [annotation['image_id'] for annotation in data['annotations']]
    unique_image_ids = list(set(image_ids))
    print("Total Images in the original dataset: {}".format(len(unique_image_ids)))

    with open(output_path, 'w') as f:
        json.dump(data, f)


def filter_val_image_with_imageids(annotation_path, output_path, image_ids):
    import json
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Filter the annotations with the image ids
    # Use try except to handle the case where the image id is not in the annotations
    for image_id in image_ids:
        try:
            data['annotations'] = [annotation for annotation in data['annotations'] if annotation['image_id'] != image_id]
        except:
            print("Image ID {} not in the annotations".format(image_id))
            continue

    unique_image_ids = list(set([annotation['image_id'] for annotation in data['annotations']]))
    print("Total Images in the filtered dataset: {}".format(len(unique_image_ids)))

    with open(output_path, 'w') as f:
        json.dump(data, f)



def show_bbox_data_distribution(annotation_path, save_path):
    # Read in the coco annotations
    # Get the bbox data out of the annotations
    # Plot the bbox data distribution by dimensions, centre, aspect ration and area, are shown as contour plots, where each contour represents the probability 
    # mass of lying among different density levels (10%, 30%, 50%, 70% and 90%) with densities obtained via Gaussian kernel density estimation, which means 
    # generating four contour plots for dimension, centre, aspect ratio and area

    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde

    annotation_name = annotation_path.split('/')[-1].split('.')[0]

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    bbox_data = [annotation['bbox'] for annotation in data['annotations']]

    dimensions = []
    centres = []
    aspect_ratios = []
    areas = []
    for bbox in bbox_data:
        try:
            dimensions.append((bbox[2], bbox[3]))
            centres.append((bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2))
            aspect_ratios.append(bbox[2] / bbox[3])
            areas.append(bbox[2] * bbox[3])
        except ZeroDivisionError:
            print("Zero Division Error")
            continue

    # TODO 写一下这个画图函数，并不是很懂

    # Assuming dimensions, centres, aspect_ratios, and areas are lists of tuples or lists
    # Convert them to numpy arrays for processing
    dimensions = np.array(dimensions)
    centres = np.array(centres)
    aspect_ratios = np.array(aspect_ratios)
    areas = np.array(areas)

    # For each feature, plot a contour plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    titles = ['Dimensions', 'Centres', 'Aspect Ratios', 'Areas']
    for ax, data, title in zip(axs.flat, [dimensions, centres, aspect_ratios, areas], titles):
        # If the data is not 1-dimensional, we need to transpose it
        if len(data.shape) > 1:
            data = data.T

        # Calculate the Gaussian KDE
        density = gaussian_kde(data)

        # If the data is 1-dimensional, plot a 1D KDE
        if len(data.shape) == 1:
            sns.kdeplot(data, fill=True, ax=ax)
        # If the data is 2-dimensional, plot a 2D KDE
        else:
            sns.kdeplot(x=data[0], y=data[1], fill=True, levels=[0.1, 0.3, 0.5, 0.7, 0.9], ax=ax)

        ax.set_title(f'{title} Distribution')

    # Set the total title and save the plot
    fig.suptitle(f'{annotation_name}.json Bounding Box Data Distribution', fontsize=25)

    # Save the plot in the save_path
    plt.savefig(f'{save_path}/{annotation_name}.png')
    plt.show()


def filter_out_annnotation_with_image_numbers(annotation_path, save_path, image_number_txt_path):
    # Read in the coco annotations
    # Get the image numbers from the image_number_txt_path
    # Filter out the annotations with the image numbers
    # Save the filtered annotations to a new file

    import json
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    with open(image_number_txt_path, 'r') as f:
        image_numbers = f.readlines()

    image_numbers = [int(image_number.replace('\n', '')) for image_number in image_numbers]

    # Filter out the annotations with the image numbers
    filtered_annotations = []

    filtered_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] not in image_numbers]

                
    data['annotations'] = filtered_annotations

    unique_image_ids = list(set([annotation['image_id'] for annotation in data['annotations']]))
    print("Total Images in the filtered dataset: {}".format(len(unique_image_ids)))

    with open(save_path, 'w') as f:
        json.dump(data, f)



image_ids = ['8844', '9769', '17379', '35062', '48504', '58393', '59598', '66706', '74733', '76416', '97337', '98716', '110042', '111086',
             '131556', '153527', '172935', '196442', '236599', '274272', '305309', '306136', '309713', '315492', '336356', '345361',
             '355817', '359677', '369323', '369541', '381587', '385190', '391722', '395575', '397681', '407943', '423123', '424135',
             '425221', '437514', '455085', '456303', '458755', '461751', '477805', '480275', '493286', '514586', '530854', '536038',
             '538236', '541773', '554266', '557672', '568584'
             ]
# Filter the training and validation datasets
# filter_annotations(train_annotation_path, 'model_data/instances_train2017_person.json')
# filter_annotations(val_annotation_path, 'model_data/instances_val2017_person.json')

filter_val_images('model_data/instances_val2017_person.json', 'model_data/instances_val2017_person_5_filtered.json')

filter_out_annnotation_with_image_numbers('model_data/instances_val2017_person_5_filtered.json', 'model_data/instances_val2017_person_5_filtered_2.json', 'model_data/image_numbers.txt')
# show_bbox_data_distribution('model_data/instances_val2017_person_5.json', 'model_data')


# show_bbox_data_distribution('model_data/instances_val2017_person.json', 'model_data')
# show_bbox_data_distribution('model_data/instances_val2017_person_5.json', 'model_data')
# show_bbox_data_distribution(train_annotation_path, 'model_data')
# show_bbox_data_distribution(val_annotation_path, 'model_data')

# filter_val_image_with_imageids('model_data/instances_val2017_person_5.json', 'model_data/instances_val2017_person_5_image_id_filter.json', image_ids)
# filter_val_image_with_imageids('model_data/instances_val2017_person.json', 'model_data/instances_val2017_person_image_id_filter.json', image_ids)
# filter_val_image_with_imageids(val_annotation_path, 'model_data/instances_val2017_image_id_filter.json', image_ids)