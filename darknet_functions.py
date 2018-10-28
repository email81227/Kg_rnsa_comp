import cv2
import numpy as np
import pydicom

from os.path import join, exists
from tqdm import tqdm


def save_img_from_dcm(dcm_dir, img_dir, patient_id):
    img_fp = join(img_dir, "{}.jpg".format(patient_id))
    if exists(img_fp):
        return
    dcm_fp = join(dcm_dir, "{}.dcm".format(patient_id))
    img_1ch = pydicom.read_file(dcm_fp).pixel_array
    img_3ch = np.stack([img_1ch]*3, -1)

    img_fp = join(img_dir, "{}.jpg".format(patient_id))
    cv2.imwrite(img_fp, img_3ch)


def save_label_from_dcm(label_dir, patient_id, row=None, img_size=1024):
    # rsna defualt image size
    label_fp = join(label_dir, "{}.txt".format(patient_id))

    f = open(label_fp, "a")
    if row is None:
        f.close()
        return

    obj_class, top_left_x, top_left_y, w, h = row

    # 'r' means relative. 'c' means center.
    rx = top_left_x/img_size
    ry = top_left_y/img_size
    rw = w/img_size
    rh = h/img_size
    rcx = rx+rw/2
    rcy = ry+rh/2

    line = "{} {} {} {} {}\n".format(obj_class, rcx, rcy, rw, rh)

    f.write(line)
    f.close()


def save_yolov3_data_from_index(dcm_dir, img_dir, label_dir, index):
    for idx, row in tqdm(index.iterrows()):
        img_fp = join(img_dir, "{}.jpg".format(row.patientId))
        if (row.width + row.height) == 0:
            box = None
        else:
            box = (0, row.x, row.y, row.width, row.height)

        if exists(img_fp):
            save_label_from_dcm(label_dir, row.patientId, box)
            continue

        # Since Kaggle kernel have small volume (5GB ?), I didn't contain files with no bbox here.
        if row.Target == 0:
            continue
        save_label_from_dcm(label_dir, row.patientId, box)
        save_img_from_dcm(dcm_dir, img_dir, row.patientId)


def save_yolov3_test_data(test_dcm_dir, img_dir, metadata_dir, name, series):
    list_fp = join(metadata_dir, name)
    with open(list_fp, "w") as f:
        for patient_id in series:
            save_img_from_dcm(test_dcm_dir, img_dir, patient_id)
            line = "{}\n".format(join(img_dir, "{}.jpg".format(patient_id)))
            f.write(line)


def write_train_list(metadata_dir, img_dir, name, series):
    list_fp = join(metadata_dir, name)
    with open(list_fp, "w") as f:
        for patient_id in series:
            line = "{}\n".format(join(img_dir, "{}.jpg".format(patient_id)))
            f.write(line)