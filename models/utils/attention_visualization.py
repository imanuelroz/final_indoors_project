import cv2
import numpy as np
import skimage.transform
from PIL import Image
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


def save_attention_over_images(original, attention_map, output_path, upscale):

    plt.ioff()
    plt.clf()

    plt.figure()
    plt.imshow(original[0])
    alpha = skimage.transform.pyramid_expand(np.array(attention_map), upscale=256 / upscale, sigma=8)
    plt.imshow(original[0])
    plt.imshow(alpha, alpha=0.7)
    plt.gcf().savefig(output_path)


def yolo_object_attentions(attention_map, object_detection_dict, output_path=None, original_img=None):
    max_bbox = [0, 0, 0, 0]
    max_object = ''
    max_att = -np.inf
    avg_att = -np.inf
    std_att = -np.inf

    for object, items in object_detection_dict.items():
        for item in items:
            bbox = item[1]
            resized_bbox = [int(i) // 2 for i in bbox]  # remove negative values (bbox starting outside image)
            left_x = np.max([resized_bbox[0], 0])
            left_y = np.max([resized_bbox[1], 0])
            right_x = np.min([resized_bbox[0] + resized_bbox[2], 255])
            right_y = np.min([resized_bbox[1] + resized_bbox[3], 255])

            bbox_attention = attention_map[left_x:right_x, left_y:right_y]
            a = np.mean(bbox_attention)
            m = np.max(bbox_attention)
            s = np.std(bbox_attention)

            if a > avg_att:
                avg_att = a
                max_att = m
                std_att = s
                max_bbox = resized_bbox
                max_object = str(object)

    if output_path is not None and original_img is not None:
        cv2.rectangle(original_img, (max_bbox[0], max_bbox[1]), (max_bbox[0] + max_bbox[2], max_bbox[1] + max_bbox[3]),
                      (255, 0, 0), 2)
        cv2.imwrite(output_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

    return max_object, max_bbox, avg_att, max_att, std_att


def detr_object_attentions(attention_map, object_detection_dict, output_path=None, original_img=None):
    max_bbox = [0, 0, 0, 0]
    max_object = ''
    max_att = -np.inf
    avg_att = -np.inf
    std_att = -np.inf

    img = Image.open(original_img)
    print(img.size)

    print(output_path)

    for object, items in object_detection_dict.items():
        for item in items:
            bbox = item[1]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            x_ratio = 256 / img.size[0]
            y_ratio = 256 / img.size[1]

            bbox_attention = attention_map[int(x_min * x_ratio):int(x_max * x_ratio), int(y_min * y_ratio):int(y_max * y_ratio)]
            if bbox_attention.size > 0:
                a = np.mean(bbox_attention)
                m = np.max(bbox_attention)
                s = np.std(bbox_attention)

                if a > avg_att:
                    avg_att = a
                    max_att = m
                    std_att = s
                    max_bbox = [x_min, y_min, x_max, y_max]
                    max_object = str(object)


    if output_path is not None and img is not None:
        img_array = np.array(img)
        cv2.rectangle(img_array, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]),
                      (255, 0, 0), 2)
        cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    return max_object, max_bbox, avg_att, max_att, std_att
