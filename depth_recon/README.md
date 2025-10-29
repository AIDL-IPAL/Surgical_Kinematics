# Surgical Scene Perception

## Depth Perception
This section focuses on monocular relative depth estimation using models like Metric3D.

### Stereo Depth Utilities
Utilities for stereo depth estimation, including calibration & depth estimation using NVLabs FoundationStereo 
A FoundationStereo Fork compatible with minimal updates: [FoundationStereo](https://github.com/liammchugh/FoundationStereo)

### Depth Model Finetuning
Details on finetuning monocular depth estimation models for surgical scene understanding tasks.
The curated HAMLYN training dataset can be found at [HAMLYN Dataset - Temp](https://drive.google.com/file/d/1-zuzIR3QZernan6uW7b0HqfXQi0OmnX-/view?usp=drive_link)

#### Steps for Finetuning:
00. Create branch with name/__ for work description.
01. Create conda env & intall requirements.txt in scene3d folder. email liam.mchugh@columbia.edu if issues arise.
0. Add submodules (specifically Metric3D): git submodule update --init --recursive or somehting like that
1. Run download_hamlyn.py to download the zip from drive (note this is currently linked from personal drive - to download from ZKLab shared drive, permissions need to be changed.)
2. Run scene3d/model_training/Metric3D_train_main.py to unzip subfolders & begin training. Adjust batch size, lr, etc as necessary by modifying default params or directly through Metric3D_train_main.py --... form
    - data should be ..-cropped versions, maintaining consistency from FoundationStereo processing during stereo depth estimations.
3. scene3d/utils/dataprocess.py contains dataset processing scripts. If data processing issues, check there first.

#### HAMLYN Folder Structure (applicable to training)
HAMLYN/hamlyn_data
- calibration
    - 01
        - intrinsic

## Surgical Tooltip detection/tracking in 3D

3D Tooltip Detection and Tracking on the JIGSAWS Suturing Dataset (https://cirl.lcsr.jhu.edu/wp-content/uploads/2015/11/JIGSAWS.pdf)

The dataset can be found:
- On google drive at https://drive.google.com/drive/u/1/folders/1AkZIlnezbXc4fGa9qqRtIdbN0wxguash 
- On the COSMOS A100 servers (srv3) at the following location on the mounted drive:
`/root/cosmos_data/sn3007/Surgery/JIGSAWS`

#### Instructions
0. Requires the following to be installed: [Metric3D](https://github.com/YvanYin/Metric3D), [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2). Florence2 uses huggingface Transformers (https://huggingface.co/microsoft/Florence-2-large)

1. Run `Florence2_box_annotate.py` to generate bounding box annotations for the videos.

2. Run `SAM2_segment_annotate.py` to generate tooltip annotations for the left/right surgical tools, based on the bounding boxes generated in step 1.

3. Run `Metric3D_depth_annotate.py` to generate monocular relative depth estimation outputs for all videos. Depth outputs with be saved as .mp4 videos.

4. Run `estimate_camera_params.py` to estimate the rotation/translation parameters for each video using the predicted camera coordinates (x_tooltip, y_tooltip, z_rel) and the ground truth kinematic 3D coordinates (x_world, y_world, z_world) obtained from the JIGSAWS dataset annotations.

5. Run `3D_metrics_analysis.py` to calculate metrics on the predicted tooltip 3D positions for the JIGSAWS dataset.

6. `3D_visualization.ipynb` is a sample notebook that shows how to visualize the 3D predictions vs. the ground truth tooltip positions.