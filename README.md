# SEDVFT

 You need to create the annotations file in the root directory. <br/>
 Because the evaluation file is too large, you need to download it yourself.<br/>
 The models file will be published after the paper appears in the journal
![Model](/Fig1.jpg "Moedl")

## Environment setup
Clone the repository and create the `m2release` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2release
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.6 and PyTorch (>1.8.0) is required to run our code. 

## Data preparation
To run the code, annotations, evaluation tools and visual features for the COCO dataset are needed.  

Firstly, most annotations have been prepared by [1], please download [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and rename the extracted folder as annotations, please download [image_info_test2014.json](http://images.cocodataset.org/annotations/image_info_test2014.zip) and put it into annotations. 

Secondly, please download the [evaluation tools](https://pan.baidu.com/s/1vP7Mt1gLLvn4HNxxOvSYxg) (Access code: xh6e) and extarct it in the project root directory.

Then, visual features are computed with the code provided by [2]. To reproduce our result, please download the COCO features file in [ResNeXt_101/trainval](https://pan.baidu.com/s/1s4B7JCrIk7CrQoFx5WOgjQ) (Access code:bnvu) and extract it as X101_grid_feats_coco_trainval.hdf5.

#### References
[1] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
[2] Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., & Chen, X. (2020). In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   


#### Acknowledgements
Thanks Cornia _et.al_ [M2 transformer](https://github.com/aimagelab/meshed-memory-transformer),
       Zhang _et.al_ [RSTNet](https://github.com/zhangxuying1004/RSTNet), and
       Luo _et.al_ [DLCT](https://github.com/luo3300612/image-captioning-DLCT) for their open source code.
       
Thanks Jiang _et.al_ for the significant discovery in visual representation [2].
