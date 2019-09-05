# BTW with enough of this data we could train a NN tracking algorithm
import pandas as pd
import numpy as np
import argparse
import cv2

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--inputVideo",
	help="choose which file to filter")
args = vars(ap.parse_args())

# STEP 1
# frame data
filename = args['inputVideo'].split('/')[-1].split('.')[0]
frames = pd.read_csv(str('./frame_data/'+filename+'.csv'))

# STEP 2
clip = args['inputVideo']
print("INFO - clip name is " +str(clip))
vs = cv2.VideoCapture(clip)

# initialise things
writer = None
frame_count = 1
new_frames = []

# STEP 3
while True:
    grabbed, frame = vs.read()
    # if a frame was not grabbed, the stream has ended
    if not grabbed:
        break
    
    print("[INFO] - processing frame number " + str(frame_count))

    # initialize writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = vs.get(5)
        print("FPS = " + str(fps))
        writer = cv2.VideoWriter('./ground_truth_clips/' + filename \
            + '_gt' + '.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]), True)

    # STEP 4 - draw rects
    # find which row the current frame is stored in (it might not exist, in which case row = 'no match')
    row = next(iter(frames[frames['frame'] == frame_count].index), None)

    if row is not None: # ie some bb exists
        print('[INFO] - found BB for frame ' + str(frame_count))
        # bb info exists for this frame, so we draw it
        startX, startY, endX, endY = (frames.iloc[row]['x1']), (frames.iloc[row]['y1']), \
            (frames.iloc[row]['x1']+frames.iloc[row]['w']), (frames.iloc[row]['y1'] + frames.iloc[row]['h'])

        # then draw it
        cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0,255,0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(60) & 0xFF

    if key == ord('q'): # manual quit
        break

    if row is None: # ie no bb found, either needs bb, or no objects
        key = cv2.waitKey(1)
        if key == ord('n'): # right arrow -> next frame
            continue
        else: # draw bb
            drawn_bb = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            temp_dict = {
                'frame': frame_count,
                'x1': drawn_bb[0],
                'y1': drawn_bb[1],
                'w': drawn_bb[2],
                'h': drawn_bb[3]
                }
        # add dict to list of dicts
        if temp_dict['x1'] != 0 or temp_dict['y1'] != 0: 
            new_frames.append(temp_dict)

    # STEP 5 - write frame to file
    if writer is not None:
        print('[WRITER] - saving frame ' + str(frame_count))
        writer.write(frame)

    # increment frame counter
    frame_count += 1

# DataFrame of new frames from list
new_frames = pd.DataFrame(new_frames, columns=['frame', 'x1', 'y1', 'w', 'h'])

# add new frames to dataframe, sort by frame
frames = frames.append(new_frames, ignore_index = True).sort_values('frame')

# save new dataframe as csv w/ same name (replace)
new_frames.to_csv('./frame_data/human_labelled/'+filename+'_new_frames.csv', index=False)
frames.to_csv('./frame_data/'+filename+'.csv', index=False)

# release the file pointers and do some cleanup
print("[INFO] cleaning up...")
if writer is not None:
    writer.release()
vs.release()
cv2.destroyAllWindows()