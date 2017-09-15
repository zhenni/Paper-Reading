# Multi-View 3D Objection Detection Network for Autonomous Driving

tags: Deep Learning, Computer Vision, Detection, CVPR 2017
---

## Main Points:

* Goal:  
  * Input: LIDAR point cloud + RGB Images; 
  * Predict: oriented 3D bounding boxes
* Method:
  * 3D object proposal generation: bird-view representation -&gt; 3D candidate boxes
  * multi-view feature fusion

## Methods

1. 3D Point Cloud Representation
	* Bird-eye view LIDAR: __height maps, density, intensity__ 
		* 2D grid, dicretization resolution: 0.1m
		* M-slice height maps: maximum height of the points
		* density: normalized #points in one grid cell  $$min(1.0, \frac{log(N+1)}{log(64)})$$
		* intensity: reflectance value of the point which has the maximum height in each cell (not slice)
	* Front-view LIDAR: __height, distance, intensity__
		* 3D point $$(x, y, z)$$
		* $$p_{fv} = (r, c)$$,
		* $$c = \lfloor atan2 ~ (y, x) ~ / ~  \Delta \theta \rfloor$$
		* $$r = \lfloor atan2 ~ (z, \sqrt{x^2+y^2}) ~ / ~ \Delta \phi \rfloor$$
		* $$\Delta \theta$$ and $$\Delta \phi$$ : horizontal and vertical resolution 
2. 3D Proposal Network
   1. Input: bird's eye view map
   2. parameters for one 3D box \(x, y, z, l, w, h\)
3. s


## Implementation Details

1. Network architecture TODO!!!!!!!!!!!!!
2. Input Representaion
	* use front-view point cloud: [0, 70.4] x [-40, 40] meters (remove points out of image boundaries)
	* bird-eye view: discretization resolution: 0.1m -> input size: 704 x 800
	* 64-beam Velofyne : 64 x 512 front view points ????
	* RGB upscale -> shortest size is 500
3. Training ans testing procedure
	* end-to-end
	* mini-batch size : 1 , sample 128 ROIs (roughly keep 25% ROIs are positive)
	* SGD, lr=0.001, #iterations=100K => reduce => lr = 0.0001, #iterations=20K
	* Anchor: car detection: (l, w) $$\in$$ {(3.9, 1.6), (1.0, 0.6), (1.6, 3.9), (0.6, 1.0)}, h = 1.56
	* Network Architecture: 3 pooling layer, no 4th pooling; 2x deconvolution
	* IoU overlap during training: positive anchors > 0.7; negative anchors < 0.5
	* empty anchors: computer an integral image over the point occupancy map
	* for non-empty anchor: nms: nms on bv boxes; not use 3D mms; IoU thresh 0.7 for nms; top 2000 boxes for training; top 300 for testing
4. Imageset Split
	* splits data in its own way: roughly half training and half validation
	* follow KITTI difficult regime: easy, moderate, hard
5. Evaluation
	* TODO !!!!!!!!!!!!!!! 


## Faster RCNN Tips

1. https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train