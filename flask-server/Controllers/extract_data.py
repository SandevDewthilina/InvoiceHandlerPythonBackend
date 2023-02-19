from flask_restful import Resource
from flask import request, send_file
import requests
import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pdf2image import convert_from_path
import layoutparser as lp
from paddleocr import PaddleOCR, draw_ocr
import json


def convert_numpy_to_json(numpy):
    # remove the first row and first column
    new_arr = numpy[1:, 0:]

    # convert the numpy array to a list of lists
    list_of_lists = new_arr.tolist()
    heading_list = numpy[0].tolist()

    obj = {'headings': heading_list, 'content': list_of_lists}
    return obj


def scrape_table(main_dir, file_name):
    folder_path = os.path.join(main_dir, file_name)
    images = convert_from_path(folder_path)
    page_dir_path = os.path.join(main_dir, 'pages')
    for i in range(len(images)):
        file = 'page' + str(i) + '.jpg'
        image_path = os.path.join(page_dir_path, file)
        images[i].save(image_path, 'JPEG')

    png_path = os.path.join(page_dir_path, 'page0.jpg')
    # consider only one table
    image = cv2.imread(png_path)
    image = image[..., ::-1]
    # load model
    model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                          threshold=0.5,
                                          label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                                          enforce_cpu=False,
                                          enable_mkldnn=True)  # math kernel library
    # detect
    layout = model.detect(image)
    x_1 = 0
    y_1 = 0
    x_2 = 0
    y_2 = 0

    for l in layout:
        # print(l)
        if l.type == 'Table':
            x_1 = int(l.block.x_1)
            print(l.block.x_1)
            y_1 = int(l.block.y_1)
            x_2 = int(l.block.x_2)
            y_2 = int(l.block.y_2)
            break
    im = cv2.imread(png_path)
    exit_image_path = os.path.join(page_dir_path, 'ext_im.jpg')
    cv2.imwrite(exit_image_path, im[y_1:y_2, x_1:x_2])
    ocr = PaddleOCR(lang='en')
    image_cv = cv2.imread(exit_image_path)
    image_height = image_cv.shape[0]
    image_width = image_cv.shape[1]
    output = ocr.ocr(exit_image_path)[0]

    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    image_boxes = image_cv.copy()

    for box, text in zip(boxes, texts):
        cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255),
                      1)
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0),
                    1)
    detection_dir = os.path.join(page_dir_path, 'detections.jpg')
    cv2.imwrite(detection_dir, image_boxes)

    im = image_cv.copy()

    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

        cv2.rectangle(im, (x_h, y_h), (x_h + width_h, y_h + height_h), (0, 0, 255), 1)
        cv2.rectangle(im, (x_v, y_v), (x_v + width_v, y_v + height_v), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(page_dir_path, 'horiz_vert.jpg'), im)
    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    horiz_lines = np.sort(np.array(horiz_out))
    print(horiz_lines)

    im_nms = image_cv.copy()

    for val in horiz_lines:
        cv2.rectangle(im_nms, (int(horiz_boxes[val][0]), int(horiz_boxes[val][1])),
                      (int(horiz_boxes[val][2]), int(horiz_boxes[val][3])), (0, 0, 255), 1)

    imn_dir = os.path.join(page_dir_path, 'pages/im_nms.jpg')
    cv2.imwrite(imn_dir, im_nms)

    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    vert_lines = np.sort(np.array(vert_out))
    print(vert_lines)

    for val in vert_lines:
        cv2.rectangle(im_nms, (int(vert_boxes[val][0]), int(vert_boxes[val][1])),
                      (int(vert_boxes[val][2]), int(vert_boxes[val][3])), (255, 0, 0), 1)

    cv2.imwrite(imn_dir, im_nms)

    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

    unordered_boxes = []

    for i in vert_lines:
        print(vert_boxes[i])
        unordered_boxes.append(vert_boxes[i][0])

    ordered_boxes = np.argsort(unordered_boxes)
    print(ordered_boxes)

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(box_1, box_2):

        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
        if inter == 0:
            return 0

        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

        return inter / float(box_1_area + box_2_area - inter)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])

            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if (iou(resultant, the_box) > 0.1):
                    out_array[i][j] = texts[b]

    out_array = np.array(out_array)
    sample_csv_path = os.path.join(page_dir_path, 'sample.csv')
    pd.DataFrame(out_array).to_csv(sample_csv_path)

    current_bank = [''] * len(out_array[0, :])

    def empty(arr):
        for i in arr:
            if i == '':
                return True
        return False

    cleaned_array = []

    for i in range(len(out_array)):
        if not empty(out_array[i]):
            current_bank = [out_array[i][j] for j in range(len(out_array[i]))]
            cleaned_array.append(current_bank)
            not_empty = True
        else:
            for j in range(len(out_array[i])):
                current_bank[j] += ' ' + out_array[i][j]
            print('-->', current_bank)
    cleaned_array = np.array(cleaned_array)
    print(cleaned_array)
    cleaned_csv_path = os.path.join(main_dir, 'cleaned.csv')
    pd.DataFrame(cleaned_array).to_csv(cleaned_csv_path)

    return convert_numpy_to_json(cleaned_array)
    # return send_file(cleaned_csv_path)


class ExtractData(Resource):

    def post(self):
        try:
            # body data
            data = request.json
            # download the file from the url
            response = requests.get(data['url'])
            main_directory = os.path.join(os.getcwd(), 'invoices/' + str(data['upload_name']))
            pages_directory = os.path.join(main_directory, 'pages')

            try:
                os.mkdir(main_directory)
            except OSError as error:
                print(error)
            try:
                os.mkdir(pages_directory)
            except OSError as error:
                print(error)

            filepath = os.path.join(main_directory, data['filename'])

            # save the file in directory
            with open(filepath, "wb") as f:
                f.write(response.content)

            # extract data and return as csv

            return scrape_table(main_directory, data['filename'])
        except:
            return {}
