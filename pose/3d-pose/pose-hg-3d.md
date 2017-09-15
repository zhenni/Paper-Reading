# Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

tags: Deep Learning, Computer Vision, Pose Estimation, ICCV 2017

Code: [https://github.com/~xingyizhou/pose-hg-3d.](https://github.com/~xingyizhou/pose-hg-3d.)

## Approach

- 2D pose estimation module and a depth regression module
- Training set: images with 3D groundtruth in the lab + images with only 2D ground truth in the wild

### 3D depth regression module

- Integration of 2D and 3D module
- **3D geometric constraint induced loss**
  - How to deal with 2D weakly-labeled data? 
  - => a loss induced from a geometric constraint(effective regularization for depth prediction) 
  $$L_{dep}(\hat Y_{dep}|I, Y_{2D}) = \left\{ \begin{matrix} \lambda_{reg}||Y_{dep} - \hat Y_{dep}||^2, & if ~ I \in \mathcal{I}_{3D}  \\ \lambda_{geo}L_{geo}(\hat Y_{dep}|Y_{2D}),&  if ~ I \in \mathcal{I}_{2D} \end{matrix}  \right. $$
      - $$\lambda_{geo}$$ and $$\lambda_{reg}$$: corresponding loss weights