import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from darknet.build.darknet.x64.darknet import *
from darknet_functions import *
from datetime import datetime
from glob import glob
from os.path import join
from shutil import copy2
from sklearn.model_selection import train_test_split

'''
Reference
1. AlexeyAB: Darknet on Windows, 
    https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    https://github.com/AlexeyAB/Yolo_mark 
'''
# 1: Run from config setting
flag = 0

# Def variables
#   The number of objects in each of 3 [yolo]-layers:
num_class = 1
num_filter = (num_class + 5) * 3
random_stat = 123
threshold = 0.2
tr_cfg = 'rsna_yolov3_tr.cfg'
ts_cfg = 'rsna_yolov3_ts.cfg'
weight_file = tr_cfg.replace('_ts.cfg', '_' + str(100) + '.weights')

# Def dirs
path = r'D:\DataSet\RSNA'
darknet = r'D:\Develop\PyCharmEnv\Scripts\darknet\build\darknet\x64'

tr_path = join(path, 'train')
ts_path = join(path, 'test')

img_dir = join(os.getcwd(), "images")           # .jpg
label_dir = join(os.getcwd(), "labels")         # .txt
metadata_dir = join(os.getcwd(), "metadata")    # .txt
log_dir = join(os.getcwd(), "logs")
backup_dir = join(os.getcwd(), "backup")        # YOLOv3 training checkpoints will be saved here
cfg_dir = join(os.getcwd(), "cfg")              # YOLOv3 config file directory

sub_dir = join(os.getcwd(), "submissions")

# Check if exist or make one.
for directory in [img_dir, label_dir, metadata_dir, log_dir, backup_dir, cfg_dir, sub_dir]:
    if os.path.isdir(directory):
        continue
    os.mkdir(directory)

# Copy yolov3.cfg to cfg dir
'''
cfg/rsna_yolov3.cfg_train
Basically, you can use darknet/cfg/yolov3.cfg files. However it won't work for RSNA. 
you need to edit for RSNA. You can just download a cfg file I edited for RSNA with 
following wget command. I refer to the following articles for editing cfg files.
    YOLOv3 blog
    YOLOv3 paper
    how to train yolov2 blog
    darknet github issues/236
https://docs.google.com/uc?export=download&id=18ptTK4Vbeokqpux8Onr0OmwUP9ipmcYO
'''
if flag == 1:
    if not os.path.isfile(join(cfg_dir, tr_cfg)):
        copy2(join(darknet, 'cfg', 'yolov3.cfg'), join(cfg_dir, tr_cfg))

        with open(join(cfg_dir, 'yolov3.cfg'), 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('batch=1', 'batch=64')
        filedata = filedata.replace('subdivisions=1', 'subdivisions=8')
        filedata = filedata.replace('classes=80', 'classes=' + str(num_class))
        filedata = filedata.replace('filters=255', 'filters=' + str(num_filter))

        # Write the file out again
        with open(join(cfg_dir, 'yolov3.cfg'), 'w') as file:
            file.write(filedata)


    tr_labels = pd.read_csv(join(path, "stage_1_train.csv"))

    save_yolov3_data_from_index(join(path, 'train'), img_dir, label_dir, tr_labels)

    # Following lines do not contain data with no bbox
    patient_id_series = tr_labels[tr_labels.Target == 1].patientId.drop_duplicates()

    tr_set, val_set = train_test_split(patient_id_series, test_size=0.1, random_state=random_stat)
    print("The # of train set: {}.".format(tr_set.shape[0]))
    print("The # of validation set: {}.".format(val_set.shape[0]))

    # train image path list
    write_train_list(metadata_dir, img_dir, "tr_list.txt", tr_set)
    # validation image path list
    write_train_list(metadata_dir, img_dir, "val_list.txt", val_set)

    # Create file obj.names in the directory build\darknet\x64\data\, with objects names - each in new line
    # Create file obj.data in the directory build\darknet\x64\data\, containing (where classes = number of objects):
    if not os.path.isfile(join(cfg_dir, 'RSNA.data')):
        data_extention_file_path = join(cfg_dir, 'RSNA.data')
        with open(data_extention_file_path, 'w') as f:
            f.write("""classes= 1
        train  = {}
        valid  = {}
        names  = {}
        backup = {}
        """.format(join(metadata_dir, "tr_list.txt"),
                   join(metadata_dir, "val_list.txt"),
                   join(cfg_dir, 'RSNA.names'),
                   backup_dir))

# Command for training with Pre-trained CNN Weights (darknet53.conv.74)
tr_cmd = '''powershell "{} detector train {} {} {} -dont_show | tee {}"'''.\
    format(join(darknet, 'darknet.exe'),
           join(cfg_dir, 'RSNA.data'),
           join(cfg_dir, tr_cfg),
           join(darknet, 'darknet19_448.conv.23'),
           join(log_dir, "train_log.txt"))

with open(join(cfg_dir, 'RSNA.names'), "w") as names:
    print("pneumonia", file=names)

os.system(tr_cmd)

# Training Loss
iters = []
losses = []
total_losses = []

with open(join(log_dir, "train_log.txt"), 'r', encoding="utf-8", errors='ignore') as f:
    for i, line in enumerate(f):
        if "images" in line:
            iters.append(int(line.strip().split()[0].split(":")[0]))
            losses.append(float(line.strip().split()[2]))
            total_losses.append(float(line.strip().split()[1].split(',')[0]))

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
sns.lineplot(iters, total_losses, label="totla loss")
sns.lineplot(iters, losses, label="avg loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
sns.lineplot(iters, total_losses, label="totla loss")
sns.lineplot(iters, losses, label="avg loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.ylim([0, 4.05])

# Test
# Create test images
ts_set = list(set(glob(join(path, 'test', '*.dcm'))))
ts_set = pd.Series(ts_set).apply(lambda pid: pid.strip().split("\\")[-1].replace(".dcm", ""))

save_yolov3_test_data(join(path, 'test'), img_dir, metadata_dir, "ts_list.txt", ts_set)

plt.imshow(cv2.imread(join(img_dir, "{}.jpg".format(ts_set.sample(1)))))

ts_cmd = '''{} detector test {} {} {} -thresh {}'''. \
    format(join(darknet, 'darknet.exe'),
           join(cfg_dir, 'RSNA.data'),
           join(cfg_dir, ts_cfg),
           join(backup_dir, weight_file),
           threshold)
os.system(ts_cmd)

# Generating submission
cfg_path = join(cfg_dir, ts_cfg)
weight_path = join(backup_dir, weight_file)

test_img_list_path = join(metadata_dir, "ts_list.txt")
gpu_index = 0
net = load_net(cfg_path.encode(),
               weight_path.encode(),
               gpu_index)
meta = load_meta(data_extention_file_path.encode())

submit_dict = {"patientId": [], "PredictionString": []}

with open(test_img_list_path, "r", encoding="utf-8", errors='ignore') as test_img_list_f:
    # tqdm run up to 1000(The # of test set)
    for line in tqdm(test_img_list_f):
        patient_id = line.strip().split('/')[-1].strip().split('.')[0]

        infer_result = detect(net, meta, line.strip().encode(), thresh=threshold)

        submit_line = ""
        for e in infer_result:
            confi = e[1]
            w = e[2][2]
            h = e[2][3]
            x = e[2][0]-w/2
            y = e[2][1]-h/2
            submit_line += "{} {} {} {} {} ".format(confi, x, y, w, h)

        submit_dict["patientId"].append(patient_id)
        submit_dict["PredictionString"].append(submit_line)

pd.DataFrame(submit_dict).to_csv(join(sub_dir, 'submission.csv'), index=False)

# TODO: solve Avg IOU: -nan(ind) ==> https://github.com/AlexeyAB/darknet/issues/1438