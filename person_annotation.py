import json

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

# Filter the training and validation datasets
filter_annotations(train_annotation_path, 'model_data/instances_train2017_person.json')
filter_annotations(val_annotation_path, 'model_data/instances_val2017_person.json')