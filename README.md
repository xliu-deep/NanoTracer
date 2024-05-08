# NanoTracer
![image](![numpy](https://img.shields.io/badge/numpy-v1.19-blue))


> Advanced Machine Learning for Iron Oxide Nanoparticle Source Detection

## Description
- Magnetic nanoparticles (MNPs), with magnetite (Fe3O4) and maghemite (γ-Fe2O3) as the most abundant species, are ubiquitously present in the natural environment. MNPs are also among the most applied engineered nanoparticles and can be produced incidentally by various human activities. Identification of the sources of MNPs is crucial for their risk assessment and regulation, which however is still an unsolved problem. Here we report a novel approach, hierarchical machine learning-aided iron (Fe)-oxygen (O) isotopic fingerprinting, to address this problem. We found that naturally occurring, incidental, and engineered MNPs have distinct Fe-O isotopic fingerprints probably due to significant Fe/O isotope fractionation during their generation processes, which enables to establish a Fe-O isotopic library covering different sources. Furthermore, we developed a hierarchical machine learning model called `NanoTracer` that not only is capable of distinguishing the sources of MNPs but also can identify the species (Fe3O4 or γ-Fe2O3) and synthetic routes of engineered MNPs. 

![image](https://github.com/xliu-deep/NanoTracer/assets/1555415/d994d230-1193-4360-88a8-25946a99e649)

- Related research manuscript "Source recognition of magnetic iron oxide nanoparticles by hierarchical machine learning-aided iron-oxygen isotopic fingerprinting" is being prepared for publication.
  
## Documentation
![image](https://github.com/xliu-deep/NanoTracer/assets/1555415/ff2db47a-8e04-490f-ab14-45f12f4aad16)
three hierarchical classification models were employed in this study: 1) Local classifier per node (LCPN, a), where several binary classifiers are trained for each node in the hierarchy (excluding the root node). 2) Local classifier per parent node (LCPPN, b), where multiple multi-class classifiers are trained for each parent node to predict the children's nodes in the hierarchy. 3) Local classifier per level (LCPL, c), where a multi-class classifier is trained for each level in the hierarchy. Our disclosure code incorporates four implementations of hierarchical classification models for the traceability of magnetic nanoparticles.


There are files for `NanoTracer`:
```
+---—Scripts for model training, predicting and visualization
|
|   flat_model.py  # flat classification modeling
|   hiclass_model.py  # hierarchical classification modeling
|   plot_contour_LCPN.py # class boundary plot using LCPN model as an example
|   predict.py # predict the new samples
|
+---data    # training data and test data
|       X_test.txt  # 142 NMPs species from the published data
|       X_test_Troll.txt  # 76 NMPs species from the Troll et al
|       X_train.txt # 113 training samples
|       y_test_flat.txt # label data of 142 NMPs species from the published data for flat model
|       y_test_hiclass.csv # label data of 142 NMPs species from the published data for hierchical model
|       y_test_Troll_flat.txt # label data of 76 NMPs species from the Troll et al for flat model
|       y_test_Troll_hiclass.csv # label data of 76 NMPs species from the Troll et al for hierchical model
|       y_train_flat.txt  # label data of 113 training samples for flat model 
|       y_train_hiclass.csv # label data of 113 training samples for hierchical model
|
\---models
    +---flat # All combinations of hyperparameters and their corresponding results during the flat model training process.
    |       BayesionOptimazation.log 
    |
    +---LCPL # All combinations of hyperparameters and their corresponding results during the LCPL model training process.
    |       BayesionOptimazation.log
    |       level0.pickle # The model trained with the optimal combination of hyperparameters on different hierchical levels.
    |       level1.pickle
    |       level2.pickle
    |       level3.pickle
    |
    +---LCPN # All combinations of hyperparameters and their corresponding results during the LCPN model training process.
    |       BayesionOptimazation.log
    |       MgP.Engineered.Mag.EA.pickle # The model trained with the optimal combination of hyperparameters on different hierchical levels.
    |       MgP.Engineered.Mag.EP.pickle
    |       MgP.Engineered.Mag.ES.pickle
    |       MgP.Engineered.Mag.pickle
    |       MgP.Engineered.Mgh.EP.pickle
    |       MgP.Engineered.Mgh.ES.pickle
    |       MgP.Engineered.Mgh.pickle
    |       MgP.Engineered.pickle
    |       MgP.Incidental.pickle
    |       MgP.Natural.pickle
    |       MgP.pickle
    |
    \---LCPPN # All combinations of hyperparameters and their corresponding results during the LCPPN model training process.
            BayesionOptimazation.log
            Engineered.pickle # The model trained with the optimal combination of hyperparameters on different hierchical levels.
            Mag.pickle
            Mgh.pickle
            MgP.pickle
            
```

## Contact Us
`NanoTracer` is for our research publication "Source recognition of magnetic iron oxide nanoparticles by hierarchical machine learning-aided iron-oxygen isotopic fingerprinting"  

 Hang Yang, Xuezhi Yang, Qinghua Zhang, Dawei Lu, Weichao Wang, Huazhou Zhang, Yunbo Yu, Xian Liu*, Aiqian Zhang, Qian Liu*, Guibin Jiang

>Email:xianliu@rcees.ac.cn
>Email:qianliu@rcees.ac.cn

