# Probabilistic Data Association for Semantic SLAM

tags: SLAM, Semantic

## Introduction

- Motivation:
  Allows vehicles to autonomously navigate in an environment without apriori knowledge of the map and without access to independent position information
- Problem to solve:
  - (What the enviroment looks like & Where is the robots, After given a path, how does the robot go) 
  -  Sensing: How the robot measures properties of itself and its environment
- What is SLAM?
  * simultaneous localization and mapping 
  * [wiki](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping)
    - Localization: inferring location given a map 
    - Mapping: inferring a map given locations 
  * SLAM is a chicken-or-egg problem: 
    - a map is needed for localization 
    - a pose estimate is needed for mapping
- Classical solutions:
    - Landmark extraction 
    - data association 
    - State estimation 
    - state update
    - landmark update
- Data Association:
  - ascertaining which parts of one image correspond to which parts of another image, where differences are due to movement of the camera, the elapse of time, and/or movement of objects in the photos. 
  - Problems:
    - You might not re-observe landmarks every time.
    - You might observe something as being a landmark but fail to ever see it again.
    - You might wrongly associate a landmark to a previously seen landmark.
- Loop Closure
  - Loop closure is the problem of recognizing a previously visited location and updating the states accordingly.
- Limitation of traditional approaches:
  - rely on low-level geometric features- > loop closure recognition based on low-level features is often viewpoint-dependent and subject to failure in ambiguous or repetitive environments
  - object recognition methods can infer landmark classes and scales, resulting in a small set of easily recognizable landmarks, ideal for view-independent unambiguous loop closure
- Goal:  address the metric and semantic SLAM problems jointly,
  - providing a meaningful interpretation of the scene, 
  - semantically-labeled landmarks address two critical issues of geometric SLAM: data association (matching sensor observations to map landmarks) and loop closure (recognizing previously-visited locations). 
- Other approches:
  - filtering methods
  - batch methods: pose graph optimization, iterative optimization methods
  - use both spatial and semantic representation
    - For localization: incorporate semantic observations in the metric optimization  
    - realtime implementation
    - global optimization for 3d reconstruction and semantic parsing. 3d space is voxelized 
    - structure from motion
    - semantic mapping
- __Contributions:__
  1. the first to tightly couple inertial, geometric, and semantic observations into a __single optimization framework__
  2. __provide a formal decomposition of the joint metric-semantic SLAM problem__ into continuous (pose) and discrete (data association and semantic label) optimiza-tion sub-problems, 
  3.  __carry experiments on several long-trajectory real indoor and outdoor datasets__ which include odometry and visual measurements in cluttered scenes and varying lighting conditions. 

## Methods

### EM Algorithms
- Wiki: [link](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- In statistics, an expectation–maximization (EM) algorithm is an iterative method to find maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables. The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step. These parameter-estimates are then used to determine the distribution of the latent variables in the next E step 
- Coin flipping Examples: [link](https://www.nature.com/articles/nbt1406/figures/1) ![em_coin](https://media.nature.com/lw926/nature-assets/nbt/journal/v26/n8/images/nbt1406-F1.gif)

### Probabilistic Data Association in SLAM

- Formualtation of the problem
  - $$\mathcal{L} \triangleq $$

### Semantic SLAM

## Experiments

- backend: GTSAM and its iSAM2 implementation, realtime
- frontend: 
  - every 15th camera frame as a keyframe
  - ORB features
  - outlier tracks: estimating the essential matrix using RANSAC 
  - assume timeframe is short => oritation differences using gyroscope measurements => only need to estimate unit translation vector (only using two point correspondence)
  - object detector: deformable parts model detection algorithm
  - a new landmark: [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) to all known landmarks, above a threshold, initial position: camera ray with estimation of depth(median depth of all geometry feature measurements)
- in practice, solve only once for the constraints weights
- datasets:
  - experimental platform: VI-Sensor (IMU, left camera)
    1. medium length(~175m), one floor of office building, object classes(red office chairs and brown four legs chairs)
    2. long length(~625m), two different floors, object classes(red chairs and doors)
    3. several loops around one room(with vicon motion tracking system), object class(red chair)
  - KITTI (05, 06)
- Results:
  - their own indoor dataset: compared with [ROVIO](https://github.com/ethz-asl/rovio), [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
    1. Fig. 3, 4,
    2. Fig. 1, 2, 5 Fig. 6: [bag-of-words based loop closure detections](https://nicolovaligi.com/bag-of-words-loop-closure-visual-slam.html)
    3. Fig. 7, 8
  - KITTI outdooe dataset: campared to [VISO2](http://www.cvlibs.net/software/libviso/), [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
    - object: cars
    - not inertial odometry => [VISO2 odometry algorithm](http://www.cvlibs.net/software/libviso/)
 
- Future Work / Limitation:
  - estimate full pose of the semantic objects (orientation in addition to positions)
  - consider data associations for past keyframes
  - consider multiple sensors (only one camera?) 
  - consider non-stationary objects
  
  
## Slides:
[https://people.eecs.berkeley.edu/~jrs/speaking.html](https://people.eecs.berkeley.edu/~jrs/speaking.html)
0. __Motivation__: why come up this problem
1. __Problem__: What problem is the paper trying to solve? 
2. __Background__: Give background so everyone is on the same page
3. __Topics in class__: Connect ideas in paper to topics covered in class. __Sensors: gyroscope(IMU), cameras__
4. __Contributions__: What is the contribution compared to prior work?
5. __New ideas__: What can now be done that couldn’t be done before?
6. __New Methods / Approprate Tech Depth__: What new ideas enable this to be done?
7. __Experiments & Results__: What evaluation/experiments were performed?
8. __Limitation & Opinion for Effectiveness__: Give your opinion/analysis of effectiveness of proposed method

