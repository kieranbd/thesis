import pandas as pd
import argparse
import cv2

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--inputFile", required=True,
	help="choose which csv file to filter")
args = vars(ap.parse_args())

# find video clip in question
title = args['inputFile'].split('/')[-1].split('.')[0] # Bike05, Surf01, etc
video_file = title+'.mp4' # Bike05.mp4
frames = pd.read_csv(str('./frame_data/'+title+'.csv')) 

# open video stream
vs = cv2.VideoCapture('./raw_clips/' + video_file)

# find resolution of clip
grabbed, frame = vs.read()
orig_width = frame.shape[1]
orig_height = frame.shape[0]

print('[INFO] Resolution is ' + str(orig_width) + ' x ' + str(orig_height))

# scale
frames.x1 /= orig_width
frames.w /=  orig_width
frames.y1 /= orig_height
frames.h /=  orig_height

frames.to_csv('./frame_data/test_' + title + '.csv', index=False)