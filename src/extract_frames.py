import os
import cv2

BASE_PATH = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_PATH, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# extract frames from each video, only extract 30 frames over time
def extract_frames(vid_name, label, input_dir, out_dir, frame_per_second=6):
    currentframe = 0
    video = cv2.VideoCapture(os.path.join(input_dir, vid_name))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    gap = length/33
    sampling_points = [int(x*gap) for x in range(0,30)]
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    count = 0
    while(True):   
        # reading from frame 
        ret, frame = video.read() 
        if ret:
            # if video is still left continue creating images 
            name = os.path.join(out_dir, f"{vid_name[:-4]}_frame_{currentframe}_{label}.jpg")
            # writing the extracted images 
            if count < 30 and currentframe == sampling_points[count]:
                frame=cv2.resize(frame,(224,224), interpolation = cv2.INTER_AREA)
                cv2.imwrite(name, frame)
                count += 1
            currentframe += 1
        else: break
    # Release all space and windows once done 
    video.release() 
    cv2.destroyAllWindows()

#read splitting files
def read_list(input_file):
    with open(input_file) as input_text:
        return [line.strip().split() for line in input_text if line]

UCF101_DIR = os.path.join(DATA_DIR, 'UCF-101')
train_test_list_dir = os.path.join(DATA_DIR, 'ucfTrainTestlist')
train_lists = [os.path.join(train_test_list_dir, f"trainlist0{x}.txt") for x in range(1,4)]
test_lists = [os.path.join(train_test_list_dir, f"testlist0{x}.txt") for x in range(1,4)]

class_ind = os.path.join(train_test_list_dir, 'classind.txt')
temp = read_list(class_ind)
labels = []
for label in temp:
    labels.append(label[1])
del temp

#use testlist01 and trainlist01 to split data
data = read_list(train_lists[0])
class_num = 10

#data[i][0] = video name, data[i][1] = label
for i, video in enumerate(data):
    if int(video[1]) > class_num:
        break
    temp = video[0].split('/')
    label_dir = temp[0]
    video_name = temp[1]
    in_dir = os.path.join(UCF101_DIR, label_dir)
    if i % 3 == 0:
        out_dir = os.path.join(VALIDATION_DIR, label_dir, video_name[:-4])
    else:
        out_dir = os.path.join(TRAIN_DIR, label_dir, video_name[:-4])
    extract_frames(video_name, video[1], in_dir, out_dir)

data = read_list(test_lists[0])
for video in data:
    temp = video[0].split('/')
    label_dir, video_name = temp[0], temp[1]
    #label for test datum
    label = labels.index(label_dir)+1
    if label > class_num:
        break
    #input directory
    in_dir = os.path.join(UCF101_DIR, label_dir)
    #test directory
    out_dir = os.path.join(TEST_DIR, label_dir, video_name[:-4]) 
    extract_frames(video_name, label, in_dir, out_dir)