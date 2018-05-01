## Introduction

Links:

[Visual Domain Adaptation Challenge](http://ai.bu.edu/visda-2017/):
  - Segmentation Winner: [MSRA](http://ai.bu.edu/visda-2017/assets/attachments/VisDA_MSM_at_MSRA.pdf)
  - Segmentation honorable mention: [NEC](http://ai.bu.edu/visda-2017/assets/attachments/VisDA_VLLab.pdf)


1. **Feature Level**
  - align the features extracted from the networks across the sourse and the target domains. (__Unsupervised__: no labeled target samples)
  - typically, minimize some measure of the distance between the source and the target feature distribution, 
    - maximum mean discrepency: 
    - Correlation distance
    - adversarial discriminator accuracy
  - Limitaions:
    1. align marginal distributions does not enforce any semantic consistency. (e.g. car->bicycle) 
    <span style="color:red">If the feature distributions are quite different??</span>
    2. higher levels of a deep representation can fail to model aspects of low-level appearance variance __lose some low-level/local feature/information__
2. ** Pixel/Frame Level** : **Generative**
  - similar distribution alignment. Translate the source data to the style of a target domain. <span style="color:red">similar distribution alignment: If the feature distributions are quite different??</span>
  - __Unsupervised__ methods:
  - Limiation:
    - small image sizes and limited domain shifts;
    - controlled enveironment;
    - may not preserve content: crucial semantic information may be lost
- **Multilevel**


- Discriminator: 
  - cannot distinguish the segmentation result between source and target
  - 
  
- How to use the images only with the view point changes
- The distributiion of the features are very different from each other
   - Also, scale is different. 
- The categories of source and target are not exactly same.
    