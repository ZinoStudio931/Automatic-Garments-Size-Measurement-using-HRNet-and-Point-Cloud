
# Automatic Size Measurement of Garments Using Computer Vision Deep Learning Models and Point Cloud Data
Welcome to the Repository for **Automatic Garment Size Measurement models**

<!-- ![Sample results of Automatic Size Measurements](/figs/sizing-results-samples.png) -->
<p align="center">
   <img src="/figs/sizing-results-samples.png" width="100%" height="100%">    
   <em>Sample results of Automatic Size Measurements</em>
</p>


## Introduction
We devised a **novel method for measuring clothes size with a mobile device with a LiDAR and camera** like Ipad pro.
All you have to do is lay the clothes you want to measure on a flat surface, and then simply take a single capture of the clothes with your mobile device. (We submitted an article that introduces our method in detail to a journal and is currently under review.)
<!-- ![Coordinates mapping](/figs/mapping.png){: width="40" height="40"} -->
<p align="center">
   <img src="/figs/mapping.png" width="60%" height="60%">    
</p>
<p>
   <em>Mapping Image & Depth map & Point cloud</em>
</p>

In this repo, you can demonstrate the process of finding the size with the test dataset.
The test dataset consists of 330 sets of (image, depth map and point clouds), as shown in table below. 
You can download the dataset in our official homepage [HifiAI](http://www.hifiai.pe.kr/new/5/) or [Onedrive storage](https://drive.google.com/drive/folders/1ePJE7KybnQlHRa-4QxiX-jPfpIYKf_Oa?usp=sharing) 
(Point detector(: pretrained model) is available upon request (temporarily, until the article is in reviewing). Please, fill free to email the author: <seo.kim931@gmail.com>)

<!-- ![Test datasets](/figs/table1.png) -->
<p align="center">
   <img src="/figs/table1.png" width="60%" height="60%">   
</p>
<p>
   <em>Description of garments in the dataset</em>
</p>

* Our work,    
(Modified from [deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch))   
(Training dataset from [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2))   
(inspired by [HRNet for Fashion Landmark Estimation](https://github.com/svip-lab/HRNet-for-Fashion-Landmark-Estimation.PyTorch))   
(inspired by [Aggregation and Finetuning for Clothes Landmark Detection](https://github.com/lzhbrian/deepfashion2-kps-agg-finetune))   

## Main Results
### Landmark Estimation Performance on DeepFashion2 Test set
Experiment results showed that the average relative error was **less than 2%(: 1.59%)**.
This result corresponds to an error of **less than 1 cm at a length of 50 cm**.
(Further details can be found our *in-reviewing* article later.)

## Quick start
### Preparation
1. Install pytorch >= v1.2 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should follow the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as `${POSE_ROOT}`.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Create following directories at `${POSE_ROOT}` :

   ```
   cd .. 

   mkdir data
   mkdir models
   mkdir log
   mkdir output 
   ```

   After all processes, following sub-directories should be prepared:

   ```
   ${POSE_ROOT}
   |-- data
   |-- experiments
   |-- figs
   |-- lib
   |-- log
   |-- models
   |-- output
   |-- tools 
   |-- README.md
   `-- requirements.txt
   ```

### Download the Point Detector and Test Dataset

1. Prepare the Sizing Point Detector
   Place the Point Detector under `${POSE_ROOT}/models`. You can get a download link to get the pretrained models(.pth), upon request by the author's email: <seo.kim931@gmail.com>.

2. Download the Test dataset
   Download the dataset from the [Onedrive storage](https://drive.google.com/drive/folders/1ePJE7KybnQlHRa-4QxiX-jPfpIYKf_Oa?usp=sharing). Extract the dataset under `${POSE_ROOT}/data` as follow.

   ```
    ${POSE_ROOT}
    `-- data
        `-- deepfashion2
            `-- validation
                |-- image                           (330 RGB images in total)
                |-- depth                           (330 gray-scale depth images in total)
                |-- ply                             (330 point cloud data in total)
                `-- val-coco_style.json             (annotation file for test)
    ```

### Install the API
1. Clone [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) repo. (hereafter, `${DF2_ROOT}`)

    ```
    ${PARENT_ROOT}
    |-- ${POSE_ROOT}
    `-- ${DF2_ROOT}
    ```

2. Install the deepfashion_api
    Enter `${DF2_ROOT}/deepfashion2_api/PythonAPI` and run
    ```
    python setup.py install
    ```
    Note that the `deepfashion2_api` is modified from the `cocoapi` without changing the package name. Therefore, conflicts occur if you try to install this package when you have installed the original `cocoapi` in your computer. We provide two feasible solutions: 1) run our code in a `virtualenv` 2) use the `deepfashion2_api` as a local pacakge. Also note that `deepfashion2_api` is different with `cocoapi` mainly in the number of classes and the values of standard variations for keypoints.

### Testing

Note that the `GPUS` parameter in the `yaml` config file is deprecated. To select GPUs, use the environment varaible:

```bash
 export CUDA_VISIBLE_DEVICES=1
```

**Testing sizing point detector** on test dataset using pretrained model:
```bash
python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet_point-detector.pth \
    TEST.USE_GT_BBOX True
```

### Calculate the size and visualization 
Calculation and visualization modules will be released as soon as the article review is done.

## Discussion
- The article review is currently in progress. After publication, related files will be provided for testing.

