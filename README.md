# Localizing Moments in Video with Natural Language.

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Natural Language." ICCV (2017).

Find the paper [here](https://arxiv.org/pdf/1708.01641.pdf) and the project page [here.](https://people.eecs.berkeley.edu/~lisa_anne/didemo.html)

```
@inproceedings{hendricks17iccv, 
        title = {Localizing Moments in Video with Natural Language.}, 
        author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan}, 
        booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 
        year = {2017} 
}
```

License: BSD 2-Clause license

## Running the Code

This repository now uses **PyTorch** (Caffe dependency removed).

### 1) Create and use a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Download required features and embeddings

```bash
bash download/get_models.sh
```

**Evaluation**

Look at `utils/eval.py` if you want to evaluate ranked segment predictions.

1. Train RGB and flow models (or provide checkpoint tags from previous training).
2. Run:

```bash
bash test_network.sh <rgb_snapshot_tag> <flow_snapshot_tag> [iter]
```

This runs val/test for RGB and flow checkpoints and then runs late fusion.

Model snapshots are stored as:

```text
snapshots/<snapshot_tag>_iter_<iter>.pt
```

You should get the following outputs:

| | Rank@1 | Rank@5 | mIOU |
| --- | --- | --- | --- |
| RGB val | 0.2442 | 0.7540 | 0.3739 |
| Flow val | 0.2626 | 0.7839 | 0.4015 |
| Fusion val (lambda 0.5) | 0.2765 | 0.7961 | 0.4191 |
| RGB test | 0.2312 | 0.7336 | 0.3549 |
| Flow test | 0.2583 | 0.7540 | 0.3894 |
| Fusion test (lambda 0.5) | 0.2708 | 0.7853 | 0.4053 |

**Training**

Use `run_job_rgb.sh` to train an RGB model and `run_job_flow.sh` to train a flow model.

You can also run training directly, e.g.:

```bash
python build_net.py --feature_process_visual feature_process_context \
                    --loc \
                    --vision_layers 2 \
                    --language_layers lstm_no_embed \
                    --feature_process_language recurrent_embedding \
                    --visual_embedding_dim 500 100 \
                    --language_embedding_dim 1000 100 \
                    --max_iter 30000 \
                    --snapshot 10000 \
                    --stepsize 10000 \
                    --base_lr 0.05 \
                    --loss_type triplet \
                    --lw_inter 0.2 \
                    --tag rgb_model_
```

`build_net.py` writes deploy metadata to `prototxts/*.json` and checkpoints to `snapshots/*.pt`.

## Dataset

### Annotations

To access the dataset, please look at the json files in the "data" folder.  Our annotations include descriptions which are temporally grounded in videos.  For easier annotation, each video is split into 5-second temporal chunks.  The first temporal chunk correpsonds to seconds 0-5 in the video, the second temporal chunk correpsonds to seconds 5-10, etc.  The following describes the different fields in the json files:

* annotation_id: Annotation ID for description
* description: Description for a specific video segment
* video: Video name
* times: Ground truth time points marked by annotators.  The time points indicate which chunk includes the start of the moment and which chunk includes the end of the moment.  An annotation of (3,3) indicates that a moment starts at 3x5=15 seconds and ends at (3+1)x5=20 seconds.  An annotation of (1,4) indicates that a moment starts at 1x5=5 seconds and ends at (4+1)x5=20 seconds.  Note that the last segment is not always 5 seconds long.  For example, for a video which is 28.2 seconds long, the annotation (5,5) will correpsond to 5x5=25 seconds to min((5+1)x5 seconds, video length) = 28.2 seconds.  Some videos are longer than 30 seconds.  These videos were truncated to 30 seconds during annotation.
* download_link: A download link for the video.  Unfortunately, this download link does not work for many Flickr videos anymore.  See "Getting the Videos" for more details.
* num_segments:  Some videos are a little shorter than 25 seconds, so were split into five temporal chunks instead of six.

### Getting the Videos

1.  Download videos from AWS (preferred method).  YFCC100M images and videos are stored on AWS [here](https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/).  Because many videos have been deleted off of Flickr since I collected the dataset, it is best to access the videos stored on AWS instead of trying to download directly from Flickr.  To download the videos used in my dataset use the script download_videos_AWS.py:

`python download_videos_AWS.py --download --video_directory DIRECTORY`

There are 13 videos which are not on AWS which you may download from my website [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos_AWS.zip) (I don't have enough space to store all the videos on my website -- Sorry!)

2.  Download videos directly from Flickr.  This is what I did when collecting the dataset, but now many Flickr videos have been deleted and many people have had issues running my download script.  To download videos directly from Flickr:

Use the script download_videos.py:
`python download_videos.py  --download --video_directory DIRECTORY`

When I originally released the dataset, ~3% of the original videos had been deleted from Flickr.  You may access them [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos.zip).  If you find that more videos are missing, please download the videos via the AWS links above.

3.  Download from [Google Drive](https://drive.google.com/drive/u/1/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc).  

You can view the Creative Commons licenses in "video_licenses.txt".

### Pre-Extracted Features

You can access preextracted features for RGB [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5) and for flow [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5).  These are automatically downloaded in "download/get_models.sh".  To extract flow, I used the code [here](https://github.com/wanglimin/dense_flow).

I provide re-extracted features in the Google Drive above.  You can use [this script](https://github.com/LisaAnne/LocalizingMoments/blob/master/make_average_video_dict.py) to create a dict with averaged RGB features and [this script](https://github.com/LisaAnne/LocalizingMoments/blob/master/make_average_video_dict_flow.py).  The average features will be a bit different than the original release, but did not influence any trends in the results.
