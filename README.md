# Kinematics Reconstruction
## Github Structure
Our repository is organized into multiple directories for different aspects of robotic surgery research:

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

## Datasets (hyperlinks coming soon): 
