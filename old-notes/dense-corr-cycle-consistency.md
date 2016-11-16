# Learning Dense Correspondence via 3D-guided Cycle Consistency

Paper: CVPR 2016 (Oral)

Link: [https://arxiv.org/abs/1604.05383](https://arxiv.org/abs/1604.05383)

## Approach

- **Goal**: predict a dence flow/corrspondence $$F_{a,b}: \mathbb{R}^2 \rightarrow \mathbb{R}^2$$ between images $$a, b$$  
   - $$F_{a, b} = (p_x-q_x, p_y-q_y)$$ relative offset
   - $$M_{a,b}: \mathbb{R}^2 \rightarrow [0, 1]$$ matchablity, 1 if correspondence exists, 0 if not
- **Cycle-consistency**
  - training real images $$r_1, r_2$$, 3D CAD model with 2 sythetic views $$s_1, s_2$$
  - $$<s_1, s_2, r_1, r_2>$$ learn to predict flows $$F_{s_1, r_1}$$, $$F_{r_1, r_2}$$, $$F_{r_1, s_2}$$
  - $$\tilde F_{s_1 s_2}$$ :as ground-truth: provided by the rendering machine

- **Learning Dense Correspondence**
  - minimize objective function $$\displaystyle \sum_{<s_1, s_2, r_1, r_2>}\mathcal{L}_{flow}(\tilde F_{s_1, s_2}, F_{s_1, r_1}\circ F_{r_1, r_2} \circ F_{r_2, s_2})$$
  - Transitive flow composition $$\bar F_{a,c} = F_{a, b} \circ F_{b, c}$$, $$\bar F_{a,c}(p) = F_{a, b}(p) + F_{b, c}(p + F_{a, b}(p))$$ 
  - Truncated Euclidean loss $$\mathcal{L}_{flow}$$: $$\mathcal{L}_{flow}(\tilde F_{s_1, s_2}, \bar F_{s_1, s_2}) = \sum_{p|\tilde M_{s_1, s_2}(p)=1} \min (||\tilde F_{s_1, s_2}(p) - \bar F_{s_1, s_2}(p)||^2, T^2)$$
  - In experiments, $$T=15$$pixels;
  - Why truncated loss: to be more robust to spurious outliers for
training, especially during the early stage when the network
output tends to be highly noisy.

- **Learning Dense Marchability**
  - objective function: per-pixel cross-entropy loss $$\displaystyle \sum_{<s_1, s_2, r_1, r_2>}\mathcal{L}_{mat}(\tilde M_{s_1, s_2}, M_{s_1, r_1}\circ M_{r_1, r_2} \circ M_{r_2, s_2})$$
  - $$\tilde M_{r_2, s_2}$$: ground-truth matchability map
  - Matchability map composition: $$\bar M_{a,c}(p) = M_{a,b}(p) M_{b,c}(p+F_{a,b}(p))$$
  - fix $$M_{s_1,r_1}=1$$ and $$M_{r_2, s_2} = 1$$, only train the CNN to infer $$M_{r_2, s_2}$$ (Due to the multiplicative nature in matchability composition)

## Network

1. **feature encoder** of **8 convolution layers** that
extracts relevant features from both input images with
shared network weights; 
2. **flow decoder** of **9 fractionallystrided/up-sampling
convolution (uconv) layers** that assembles
features from both input images, and outputs a dense
flow field; 
3. **matchability decoder** of **9 uconv layers** that
assembles features from both input images, and outputs a
probability map indicating whether each pixel in the source
image has a correspondence in the target.


- conv+relu(except last uconv for decoders)
- kernel 3*3
- no pooling; stride = 2 when in/decrease the spatial dimension
- output of matchability decoder + sigmoid for normalization
- training: same network for $$s_1 \rightarrow r_1, r_1 \rightarrow r_2, s_1 \rightarrow r_2$$

## Experiments

### Training set

- real images:  PASCAL3D+ dataset 
  - cropped from bounding box;
  - rescaled to 128*128
- 3D CAD models: ShapeNet database
  - render 3D models from the same viewpoint
  - choose K=20 nearest models using HOG Euclidean distance
- valid training quartet for each category: 80,000

### Network training

- Initialization: 
  - feature encoder + flow decoder pathway: mimic SIFT flow by randomly sampling image pairs from the training quartets and training the network to minimize the Euclidean loss between the network prediction and the SIFT flow output on the sampled pair
  - other initialization strategies (e.g. predicting ground-truth flows between synthetic images), and found that initializing with SIFT flow output works the best.
- Parametes:
  - ADAM solver $$\beta_1$$ = 0.9, $$\beta_2$$ = 0.999, lr = 0.001,
step size of 50k, step multiplier of 0.5 for 200k iterations.
- batch = 40 during initialization and 10 quartets during fine-tuning.

### Feature 

embedding layout appears to be viewpoint-sensitive
(might implicitly learn that viewpoint is an important cue for correspondence/matchability tasks through our consistency training.)

### Keypoint transfer task
Evaluate the quality ofcorrespondence output

- For each category, we exhaustively sample pairs from the val split (not seen during training), and determine if a keypoint in the source image is transferred correctly (by measuring the Euclidean distance between our correspondence prediction and the annotated ground-truth (if exists) in the target image)
- . A correct transfer: prediction falls within $$\alpha \cdot \max(H, W)$$ pixels of the ground-truth with H and W being the image height and width, respectively (both are 128 pixels in our case)
- Metric: e percentage of correct keypoint transfer (PCK)

### Matchability prediction

- PASCAL-Part dataset(provides humanannotated part segment labeling)

###  Shape-to-image segmentation transfer


  





