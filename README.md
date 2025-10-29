# ProjectSurgeryHernia

## Github Structure
Our repository is organized into multiple directories for different aspects of robotic surgery research:

- **Applications**: [Spatial Reconstruction](scene3d) [Pose Estimation / Analytics](pose_analytics), [Gesture Recognition](gestureRecognition), [Phase Segmentation](SurgeryHerniaRealTime), [Tooltip Tracking](tooltip_tracking), [Objective Performance Indicators (OPI) Dashboard](OPI_dashboard)
- **AI/ML Submodules**: [Depth-Anything-V2](Depth-Anything-V2), [FoundationStereo](FoundationStereo), [Metric3D](Metric3D), [monst3r](monst3r), [sam2](external/sam2), [resnet](resnet), [Trans-SVNet](Trans-SVNet), [transformer](transformer)
- **Development**: [sandboxes](sandboxes), [drylab](drylab), [TransferLearning](TransferLearning)

Repository also contains generalizable ML tools like data-distributed training and visualizations in scene3d & pose_analytics (details in folder-specific READMEs)

## Valuable Links
#### Project report links:
* [Generalized Depth Estimation for Laparoscopic Surgery] (https://docs.google.com/document/d/1oaPYvhEME6rahrYLCgC47wDzbtJajee4hTOMC6zDFiI/edit?usp=sharing)
* [Behavioral Kinematic Analytics in Tele-Operated Robotic Surgery] (https://docs.google.com/document/d/1k7zhk_QSCqjP2cHTlfGgAeLrxz392kvZ66-qPg3Chuo/edit?usp=sharing)
* Google liondrive link: https://drive.google.com/drive/folders/1veH4NPo8Sc6UhLw2IdjjIcLcMW5onWZW?usp=sharing
* Grouper group: eeZKlab_black eezklab_black@columbia.edu
* Slack: surgery_hernia

## Datasets (Check project folders for specific preprocessing scripts): 
[HAMLYN](datasets/HAMLYN), [JIGSAWS](datasets/JIGSAWS), [cholect50](datasets/cholect50), [surgpose](datasets/surgpose), [surgvu](datasets/surgvu), [Hernia Preprocessing](hernia_preprocessing)
- GCS Bucket: [**columbia-ecmb6691 (gs://columbia-ecmb6691)**](https://console.cloud.google.com/storage/browser/columbia-ecmb6691)

## Scene Reconstruction & Pose Analytics Pipeline
Our scene reconstruction and pose analytics pipeline follows these key stages:

1. **YOLO-pose Pretraining / Finetuning**: Initial model training using the SurgPose dataset to establish foundational pose recognition capabilities. Further refinement of the pretrained model using SurgVU dataset pose annotations (Northwell Physicians Group, Encord. Contact liam.mchugh@columbia.edu) for domain-specific adaptation.

3. **Kinematic Inference**:
   - **Core Pose Detection**: Extraction of key instrument positions and orientations
   - **Optional Enhancements**:
   - Stereo/monocular depth inference for enhanced spatial awareness
   - SAM instrument masking for constraining x/y & especially depth projections

4. **Kinematic Clustering**: Analysis of movement patterns to identify surgical gestures and techniques

![Inference Diagram](docs/images/kinematic_inference_schematic.png)

## *Internal* Transfer Data ([`gsutil`](https://cloud.google.com/storage/docs/gsutil))

- To download all videos, run
  ```
  gsutil -mq cp gs://columbia-ecmb6691/surgery.videos.hernia/* [SOME_DIRECTORY]
  ```
- To upload the dataset folder (with all its subfolders), run
  ```
  gsutil -mq cp [SOME_DIRECTORY]/** gs://columbia-ecmb6691/surgery.dataset
  ```

## 2022 Spring E6691 course projects on Surgery for Hernia
* https://github.com/ecbme6040/e6691-2022spring-assign2-ky2446-ky2446
* https://github.com/ecbme6040/e6691-2022spring-assign2-HAAV-av3023-ha2605
* https://github.com/ecbme6040/e6691-2022spring-assign2-yhyh-yj2677-hl3515-jw4167
* https://github.com/ecbme6040/e6691-2022spring-assign2-CHYF-CH3370-YF2578
* https://github.com/ecbme6040/e6691-2022spring-assign2-VCSZ-yc3998-fs2752-cz2678
* https://github.com/ecbme6040/e6691-2022spring-assign2-CCLZ-lz2814-cc4718-gs3160
* https://github.com/ecbme6040/e6691-2022spring-assign2-FAJN-fm2725-ak4592-jn2515
* https://github.com/ecbme6040/e6691-2022spring-assign2-tgsf-tjg2148-sf3043
* https://github.com/ecbme6040/e6691-2022spring-assign2-SURG-an3078-bmh2168-wab2138
* https://github.com/ecbme6040/e6691-2022spring-assign2-CWLT-jl5999-yc3840-yw3747
* https://github.com/ecbme6040/e6691-2022spring-assign2-ZOJP-zo2151-jp4201
* https://github.com/ecbme6040/e6691-2022Spring-assign2-CELB-bl2899-ec3576
