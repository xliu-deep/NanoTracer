# NanoTracer

> Advanced Machine Learning for Iron Oxide Nanoparticle Source Detection

## Description
- Magnetic nanoparticles (MNPs), with magnetite (Fe3O4) and maghemite (γ-Fe2O3) as the most abundant species, are ubiquitously present in the natural environment. MNPs are also among the most applied engineered nanoparticles and can be produced incidentally by various human activities. Identification of the sources of MNPs is crucial for their risk assessment and regulation, which however is still an unsolved problem. Here we report a novel approach, hierarchical machine learning-aided iron (Fe)-oxygen (O) isotopic fingerprinting, to address this problem. We found that naturally occurring, incidental, and engineered MNPs have distinct Fe-O isotopic fingerprints probably due to significant Fe/O isotope fractionation during their generation processes, which enables to establish a Fe-O isotopic library covering different sources. Furthermore, we developed a hierarchical machine learning model called `NanoTracer` that not only is capable of distinguishing the sources of MNPs but also can identify the species (Fe3O4 or γ-Fe2O3) and synthetic routes of engineered MNPs. 

![image](https://github.com/xliu-deep/NanoTracer/assets/1555415/d994d230-1193-4360-88a8-25946a99e649)

- Related research manuscript "Source recognition of magnetic iron oxide nanoparticles by hierarchical machine learning-aided iron-oxygen isotopic fingerprinting" is being prepared for publication.
  
## Documentation
![image](https://github.com/xliu-deep/NanoTracer/assets/1555415/ff2db47a-8e04-490f-ab14-45f12f4aad16)
three hierarchical classification models were employed in this study: 1) Local classifier per node (LCPN, a), where several binary classifiers are trained for each node in the hierarchy (excluding the root node). 2) Local classifier per parent node (LCPPN, b), where multiple multi-class classifiers are trained for each parent node to predict the children's nodes in the hierarchy. 3) Local classifier per level (LCPL, c), where a multi-class classifier is trained for each level in the hierarchy. Our disclosure code incorporates four implementations of hierarchical classification models for the traceability of magnetic nanoparticles.


There are instructions for `NanoTracer`:

```
|   flat_model.py
|   hiclass_model.py
|   plot_contour_LCPN.py
|   predict.py
|
+---data
|       X_test.txt
|       X_test_Troll.txt
|       X_train.txt
|       y_test_flat.txt
|       y_test_hiclass.csv
|       y_test_Troll_flat.txt
|       y_test_Troll_hiclass.csv
|       y_train_flat.txt
|       y_train_hiclass.csv
|
\---models
    +---flat
    |       BayesionOptimazation.log
    |
    +---LCPL
    |       BayesionOptimazation.log
    |       level0.pickle
    |       level1.pickle
    |       level2.pickle
    |       level3.pickle
    |
    +---LCPN
    |       BayesionOptimazation.log
    |       MgP.Engineered.Mag.EA.pickle
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
    \---LCPPN
            BayesionOptimazation.log
            Engineered.pickle
            Mag.pickle
            Mgh.pickle
            MgP.pickle
            
```

## Contact Us

Email:xianliu@rcees.ac.cn
Email:qianliu@rcees.ac.cn

