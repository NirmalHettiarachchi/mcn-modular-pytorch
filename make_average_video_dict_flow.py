import os
import numpy as np
import h5py
import pdb
import sys
import pickle as pkl

feature_root = 'flow_features/' #wherever your features are
video_list = [feature_root + v for v in os.listdir(feature_root)]


def load_fps_dict(path):
    with open(path, 'rb') as handle:
        raw = handle.read()
    try:
        return pkl.loads(raw, encoding='latin1')
    except Exception:
        # Windows CRLF conversion can corrupt this Python2-text pickle.
        return pkl.loads(raw.replace(b'\r\n', b'\n'), encoding='latin1')


fps_dict = load_fps_dict('data/frame_rate_clean.p')

def make_h5_dict(name):

    seconds_per_chunk = 5

    np_data = np.load(video_list[-1])
    feature_dim = np_data['global_pool'].shape[2]
 
    feature_dict = {}
    for key in fps_dict.keys():
        feature_dict[key] = np.zeros((30 // seconds_per_chunk, feature_dim))

    for i, video in enumerate(video_list):
         sys.stdout.write('\r%d/%d' %(i, len(video_list)))
         video_name = video.split('global_')[-1].split('.npz')[0]
         average_frames = feature_dict[video_name]
         np_data = np.load(video)
         features = np_data['global_pool'] 

         
         #subsample depends on fps of original video (did not extract at constant frame rate) 
         fps = fps_dict[video_name]
         #samples every 5 flow frames when extracting features
         #frames_per_chunk = seconds_per_chunk * (fps / 5) = 5 * (fps / 5)
         frames_per_chunk = fps 
         
         count = 0
         for i in range(min(6, int(features.shape[0]/(fps+0.01))+1)):
             start = int(i*frames_per_chunk)
             end = int((i+1)*frames_per_chunk)
             average_frames[count, :] = np.mean(features[start:end, :], axis = 0).squeeze()
             count += 1
 
         feature_dict[video_name] = average_frames
 
    print("\n")
    f = h5py.File('data/%s.h5' %name, "w")
    for key in feature_dict.keys():
        f.create_dataset(key, data=feature_dict[key])
    f.close()

make_h5_dict('average_flow_feats')
