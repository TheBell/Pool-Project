import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from yolo_detector import YoloDetector
from track_balls import BallTracker

weights_file = "yolo/tiny-yolo-custom_final.weights"
cfg_file = "yolo/tiny-yolo-custom.cfg"
labelsPath = "yolo/custom.names"
labels = open(labelsPath).read().splitlines()

yolo = YoloDetector(cfg_file, weights_file, labels, conf_level=.7)

out_dir = "Data"
video_dir = "VideosHD"
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

def process_video(file, start_frame=500, display_video=False):
    cap = cv2.VideoCapture(file) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bt = BallTracker(yolo, start_frame)

    while True:
        if current_frame % 10000 == 0:
            print(f"Proccessing frame {current_frame} / {total_frames}")

        ret, frame = cap.read()

        if ret:
            # Only process every 5 frames
            if current_frame % 5 == 0:
                img = bt.update(frame, draw_balls=display_video)
                # Display video for demonstration
                if display_video:
                    cv2.imshow('Video', img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        else:
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    bt.save(os.path.join(out_dir, file[:-4] + '.csv'))
    
start_vid = 0
for i, file in enumerate(video_files[start_vid:], start_vid):
    print(f"Starting video {i+1} / {len(video_files)}")
    process_video(file, display_video=False)