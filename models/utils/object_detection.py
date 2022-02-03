import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from collections import defaultdict
import torch
from torch import nn
from torchvision.models import resnet50
import cv2
import io
import base64
matplotlib.use("TKAgg")

import numpy as np

torch.set_grad_enabled(False)


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


class ObjectDetector:
    # COCO classes
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    def __init__(self):
        # self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.model = DETRdemo(num_classes=91)
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu', check_hash=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def detect(self, im):
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0)

        # propagate through the model
        outputs = self.model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        return probas[keep], bboxes_scaled

    def plot_results(self, pil_img, prob, boxes):
        plt.ioff()
        plt.clf()

        plt.figure(figsize=(10, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, self.COLORS * 100):
            #if str(self.CLASSES[p.argmax()])=='bed':
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{self.CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')

        file_path = "static/uploads/detection.jpg"

        plt.gcf().savefig(file_path)

        img = Image.open(file_path)

        return img  # Image.fromarray(img)

    def get_detections(self, img_path):
        im = Image.open(img_path)
        scores, boxes = self.detect(im)
        objects_dict = self.get_object_dict(scores, boxes)
        detected_img = self.plot_results(im, scores, boxes)  # edit: was without assignment

        return detected_img  # objects_dict

    def get_object_dict(self, scores, boxes):
        objects_dict = defaultdict(list)

        for p, bbox in zip(scores, boxes):
            cl = p.argmax()
            label = self.CLASSES[cl]
            item = []
            item.append(p[cl])  # add score
            item.append(bbox)
            objects_dict[label].append(item)

        return objects_dict

    def detr_object_attentions(self, attention_map, object_detection_dict, original_img=None):
        max_bbox = [0, 0, 0, 0]
        max_object = ''
        max_att = -np.inf
        avg_att = -np.inf
        std_att = -np.inf

        img = original_img

        for object, items in object_detection_dict.items():
            for item in items:
                print("item", item)
                bbox = item[1]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]

                x_ratio = 256 / img.size[0]
                y_ratio = 256 / img.size[1]

                bbox_attention = attention_map[int(x_min * x_ratio):int(x_max * x_ratio),
                                 int(y_min * y_ratio):int(y_max * y_ratio)]
                if bbox_attention.size > 0:
                    a = np.mean(bbox_attention)
                    #a = np.sum(bbox_attention)
                    m = np.max(bbox_attention)
                    s = np.std(bbox_attention)

                    if a > avg_att:
                        print("Attention", a)
                        avg_att = a
                        max_att = m
                        std_att = s
                        max_bbox = [x_min, y_min, x_max, y_max]
                        max_object = str(object)

        return max_object, max_bbox, avg_att, max_att, std_att

    def mask_img(self, img_path, avg_attention):
        img = cv2.imread(img_path)
        im = Image.open(img_path)
        scores, boxes = self.detect(im)
        objects_dict = self.get_object_dict(scores, boxes)
        #max_area = 0

        cv2.imwrite('static/tmp/average_attention.png', avg_attention)

        max_object, max_bbox, avg_att, max_att, std_att = self.detr_object_attentions(attention_map=avg_attention,
                                                                                      object_detection_dict=objects_dict,
                                                                                      original_img=im)

        (xmin, ymin, xmax, ymax) = max_bbox
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)

        mask = np.zeros(shape=img.shape, dtype="uint8")

        # Draw a bounding box.
        # Draw a white, filled rectangle on the mask image
        cv2.rectangle(img=mask,
                     pt1=start_point, pt2=end_point,
                     color=(255, 255, 255),
                     thickness=-1)

        # Apply the mask and display the result
        maskedImg = cv2.bitwise_and(src1=img, src2=mask)

        cv2.imwrite('static/tmp/masked.png', maskedImg)


