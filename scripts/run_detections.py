import supervision as sv
import numpy as np
from ultralytics import YOLO
from pprint import pprint
import argparse
import csv
import os

#Get filename from command line arguments
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("input_dir")
args=parser.parse_args()
input_dir = args.input_dir




model = YOLO("inference/full_deer_dataset_60_epochs_large_model.pt")
classes_of_interest = [0,14,15,16,17,18,19,20,21,22,23]
confidence_by_class = [0.2,0.2,0.2,0.2,0.2,0.4,0.2,0.2,0.2,0.2,0.2]

#statistical analysis class
class Statistics:
    def __init__(self, min_num_animals_detected_single_frame, min_percent_frames_with_detections, min_num_consecutive_detections, min_bounding_box_size, movement_threshold):
        #Tracked
        self.first_deer_position = []
        self.last_deer_position = []
        self.frame_counter = 0
        self.max_num_animals_detected_single_frame = 0
        self.num_frames_with_detections = 0
        self.max_num_consecutive_detections = 0
        self.num_consecutive_detections = 0
        self.max_bounding_box_size = (0,0)
        self.classification = "animal"
        self.deer_detections = 0
        self.deer_moving = False
        #Thresholds
        self.min_num_animals_detected_single_frame = min_num_animals_detected_single_frame
        self.min_percent_frames_with_detections = min_percent_frames_with_detections
        self.min_num_consecutive_detections = min_num_consecutive_detections
        self.min_bounding_box_size = min_bounding_box_size
        self.min_percent_frames_with_deer = 0.08
        self.movement_threshold = movement_threshold



    def print_stats(self):
        print("max_num_animals_detected_single_frame: " + str(self.max_num_animals_detected_single_frame))
        print("percent_frames_wth_detections: " + str(self.num_frames_with_detections/self.frame_counter))
        print("num_consecutive_detections: " + str(self.max_num_consecutive_detections))
        print("bounding box size: " + str(self.max_bounding_box_size))
        print("first deer position: " + str(self.first_deer_position))
        print("last deer position: " + str(self.last_deer_position))
        print("is deer moving: " + str(self.deer_moving))
        print("classification:" + str(self.classification))

    #define the string printer for this strcuture for readable output
    def __str__(self):
        return str(pprint(vars(self)))

#check if the object is valid based on the class of the object being detected
def is_valid_object(class_id, conf):
    if class_id in classes_of_interest:
        index = classes_of_interest.index(class_id)
        if conf >= confidence_by_class[index]:
            return True
    return False


#track video level stats on a per frame basis
def statistic_tracker(detections, stat: Statistics):
    labels = []
    stat.frame_counter += 1
    for xyxy, _, confidence, class_id, _ in detections:
        #Check if each detection is valid using the helper function
        if(is_valid_object(class_id, confidence)):
            if(class_id == 18): stat.deer_detections += 1
            #Only add valid detections to the list
            labels.append(f"{model.names[class_id]} {confidence:0.2f}")

            x_size = abs(xyxy[2]-xyxy[0])
            y_size = abs(xyxy[3]-xyxy[1])
            if(x_size > stat.max_bounding_box_size[0] and y_size > stat.max_bounding_box_size[1]):
                stat.max_bounding_box_size = (x_size, y_size)
            if len(stat.first_deer_position) == 0:
                stat.first_deer_position = xyxy
            stat.last_deer_position = xyxy

    if(len(labels) > 0):
        if(abs(stat.last_deer_position[0] - stat.first_deer_position[0]) > stat.movement_threshold):
            stat.deer_moving = True
        else: stat.deer_moving = False
        stat.num_frames_with_detections += 1
        stat.num_consecutive_detections += 1
        if(stat.num_consecutive_detections > stat.max_num_consecutive_detections):
            stat.max_num_consecutive_detections = stat.num_consecutive_detections
        if(len(labels) > stat.max_num_animals_detected_single_frame):
            stat.max_num_animals_detected_single_frame = len(labels)
    else:
        stat.num_consecutive_detections = 0



#after the full video is processed, logic for determining if it is a true positive video
def is_valid_video(stat: Statistics) -> bool:
    if(stat.num_frames_with_detections < stat.min_percent_frames_with_detections * stat.frame_counter):
        return False
    else:
        #Update the classification of the video
        if(stat.deer_detections >= stat.min_percent_frames_with_deer * stat.frame_counter):
    #def print_stats(self):
#        print("max_num_animals_detected_single_frame: " + str(self.max_num_animals_detected_single_frame))
#        print("percent_frames_wth_detections: " + str(self.num_frames_with_detections/self.frame_counter))
#        print("num_consecutive_detections: " + str(self.max_num_consecutive_detections))
#
            stat.classification = "deer"
    if(stat.max_num_consecutive_detections < stat.min_num_consecutive_detections): return False
    if(stat.max_bounding_box_size[0] < stat.min_bounding_box_size[0] or stat.max_bounding_box_size[1] < stat.min_bounding_box_size[1]): return False
    if(stat.max_num_animals_detected_single_frame < stat.min_num_animals_detected_single_frame): return False
    #if(stat.deer_moving == False): return False
    return True



def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)
    labels = []
    #labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    #Update statistics and procure labels
    labels = statistic_tracker(detections, g_stat)
    #TODO: Only draw boxes if they satisfy detection confidence and size
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame




def process_video(current_video):
    VIDEO_PATH = current_video
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    #TODO: get frame rate from video img_info
    #TODO: set Statistics frame count for threshold to be 3 seconds time frame rate


    sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
    if(is_valid_video(g_stat) == False):
        g_stat.classification = "no animal"
    add_metadata(g_stat, current_video)
    print(current_video)
    g_stat.print_stats()


def create_metadata_file():
    if(os.path.isfile("data/metadata.csv")):
        os.remove("data/metadata.csv")

    metadata = open("data/metadata.csv", "x")
    with open('data/metadata.csv', 'w') as file:
        writer = csv.writer(file)
        field = ["filename", "number of animals","video classification"]
        writer.writerow(field)




def add_metadata(stat: Statistics, video):
    with open('data/metadata.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([video, str(stat.max_num_animals_detected_single_frame), str(stat.classification)])


create_metadata_file()

#Iterate through directory and process videos
for filename in os.listdir(input_dir):
    f = os.path.join(input_dir, filename)
    #Parameters in order to Statistics(a,b,c,d) constructor
    #a - animals per frame - the max number of animals per frame on a cumulative basis
    #b - percent of frames with detections - the total percentage of frames in the video snipper for a valid detection.
    #c - consecutive detections - at least these many frames consecutively for it so be a valid detection
    #d - bounding box size - anything smaller than this means it is a false positive
    #e - movement threshold - number of pixels an animal needs to move to be marked as a true positive
    g_stat = Statistics(1, 0.2, 30, (20,20), 10)
    process_video(f)
