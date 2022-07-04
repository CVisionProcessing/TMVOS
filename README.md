# TMVOS


![image](model.png)  
  
  Video object segmentation (VOS) is a critical yet challenging task in video analysis. Recently, many pixel-level matching VOS methods have achieved an outstanding perfor-mance without significant time consumption in fine tuning. However, most of these methods pay little attention to i) matching background pixels and ii) optimizing discriminable embeddings between classes. To address these issues, we propose a new end-to-end trainable method, namely Triplet Matching for efficient semi-supervised Video Object Segmentation (TMVOS). In particular, we devise a new triplet matching strategy that considers both the foreground and background matching and pulls the nearest negative embedding further than the nearest positive one for every anchor. As a result, this method implicitly enlarges the distances between embeddings of different classes and thereby generates accurate matching maps. Additionally, a dual decoder is applied for optimizing the final segmentation so that the model better fits for the complex background and relatively simply targets.Extensive experiments demonstrate that the proposed method achieves the outstanding performance in terms of accuracy and running-time compared with the state-of-the-art methods.  
  
Some video segmentation results:


![image](result1.png)
    
      
# Cite
Please cite our paper when you use this dataset. For more details about this paper, please refer to our paper "TMVOS: Triplet Matching for Efficient Video Object Segmentation" (https://www.sciencedirect.com/science/article/pii/S0923596522000947)

[Plain Text]
-------------
    Jiajia Liu, Hong-Ning Dai, Guoying Zhao, Bo Li, Tianqi Zhang, TMVOS: Triplet Matching for Efficient Video Object Segmentation, Signal Processing: Image Communication, Volume 107, 2022, 116779, ISSN 0923-5965.



[BibTex]
-------------
    @article{LIU2022116779,
            author = {Jiajia Liu and Hong-Ning Dai and Guoying Zhao and Bo Li and Tianqi Zhang},
            title = {TMVOS: Triplet Matching for Efficient Video Object Segmentation},
            journal = {Signal Processing: Image Communication},
            volume = {107},
            pages = {116779},
            year = {2022},
            issn = {0923-5965},
            doi = {https://doi.org/10.1016/j.image.2022.116779},
            url = {https://www.sciencedirect.com/science/article/pii/S0923596522000947},
            keywords = {Video object segmentation, Embedding learning, Triplet matching},
    }

Master
=

--model  
-------------

Model : including TMVOS model (MPS) and residual network (RESNET)  


--src  
-------------
config : Parameter setting  

dataset_utils: dataset  

main ：Training procedures  

output_result： test  

plot_tSNE: dimension reduction and visualization  

  
    
    

Use  
=

Training：  
--------
python main.py  


Evaluation:  
--------
python output_result.py  


Model:
--------
We provide trained models (https://drive.google.com/drive/folders/1bweLk5CNnHB6E8KC-CpxyNm6eEGM__fZ?usp=sharing)


The test results of this model on davis-2017：


![image](result2.png)

