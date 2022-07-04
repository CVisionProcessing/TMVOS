# TMVOS


![image](model.png)  
  
  Video object segmentation (VOS) is a critical yet challenging task in video analysis. Recently, many pixel-level matching VOS methods have achieved an outstanding perfor-mance without significant time consumption in fine tuning. However, most of these methods pay little attention to i) matching background pixels and ii) optimizing discriminable embeddings between classes. To address these issues, we propose a new end-to-end trainable method, namely Triplet Matching for efficient semi-supervised Video Object Segmentation (TMVOS). In particular, we devise a new triplet matching strategy that considers both the foreground and background matching and pulls the nearest negative embedding further than the nearest positive one for every anchor. As a result, this method implicitly enlarges the distances between embeddings of different classes and thereby generates accurate matching maps. Additionally, a dual decoder is applied for optimizing the final segmentation so that the model better fits for the complex background and relatively simply targets.Extensive experiments demonstrate that the proposed method achieves the outstanding performance in terms of accuracy and running-time compared with the state-of-the-art methods.  
    
      
      
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

