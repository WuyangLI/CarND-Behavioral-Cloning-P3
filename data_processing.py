import numpy as np
from PIL import Image

import csv


IMG_DIR = './IMG/'


def flip_img(img_dir, org_name, org_image, org_angle, rows):
    flipped_img_name = img_dir + 'FLIP_' + org_name
    flipped_img = org_image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(flipped_img_name, 'jpeg')
    rows.append([flipped_img_name, -org_angle])


def augment_data(log_path, img_dir, correction):
    samples = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    rows = []
    for sample in samples:
        center_name = sample[0].split('/')[-1]
        left_name = sample[1].split('/')[-1]
        right_name = sample[2].split('/')[-1]

        center_file_name = img_dir + center_name
        left_file_name = img_dir + left_name
        right_file_name = img_dir + right_name

        center_angle = float(sample[3])
        factor = 1.0
        if abs(center_angle) > 0.1:
            factor = 1.5
        else:
            factor = 1.0

        left_angle = center_angle + factor*correction
        right_angle = center_angle - factor*correction

        rows.append([center_file_name, center_angle])
        rows.append([left_file_name, left_angle])
        rows.append([right_file_name, right_angle])

        center_image = Image.open(center_file_name, mode='r')
        left_image = Image.open(left_file_name, mode='r')
        right_image = Image.open(right_file_name, mode='r')

        flip_img(img_dir, center_name, center_image, center_angle, rows)
        flip_img(img_dir, left_name, left_image, left_angle, rows)
        flip_img(img_dir, right_name, right_image, right_angle, rows)

    with open('./processed_log.csv', 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(rows)


if __name__=="__main__":
    test_log = './driving_log.csv'
    test_img_dir = './IMG/'
    augment_data(test_log, test_img_dir, 0.3)
