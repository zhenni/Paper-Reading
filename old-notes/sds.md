# Simultaneous Detection and Segmentation

tags: Deep Learning, Computer Vision, Segmentation, Detection, ECCV 2014

Paper: [http://www.eecs.berkeley.edu/Research/Projects/CS/vision/papers/BharathECCV2014.pdf](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/papers/BharathECCV2014.pdf)

## Task explanation
> Detect all instances of a category in an image and, for each instance, mark the pixels that belongs to it.

## Algorithm Steps

1. Proposal generation
2. Feature extraction
3. Region classification
4. Region refinement

## Evaluation Metrics

__Output__: a set of object hypotheses, with a predicted segmentation and a score. 
Correct hypothesis: its segmentation overlaps with segmentation of a ground truth instance by more than 50%. 
Penalize duplicates (as traditional bounding box task)

__Metric__:
__PR__: precision recall curve
__AP__: average precision (area under the curve)

__Notation__:
$$\text{PR}^{r}$$: computed in SDS way. (Messures accuracy of segmentation and need to get each instance separately and completely)
$$\text{PR}^{b}$$: computed in classical bounding box task

$$\text{AP}_{vol}$$: the volumn under $$\text{PR}$$ surface varying the threshold(accurate segmentation)

### Proposal generation

Care about segments, not just boxes.
Use __MCG__: outperform on the object level Jaccard index metric.(measures the average best overlap achieved by a candidate for ground truth object)
$\text{AP}^{b}$ improved ~ 0.7% if use MCG instead of Selective Search

### Feature extraction 

- start from R-CNN (object detector): 
 - Finetuning: take bounding boxes from Selective Search -> paded -> crop -> warp to a square -> feed to network
 Positives: bounding boxes that overlap with GT by more than 50%, label: GT's label that overlap the most with the box
 - Test: b-box->pad->crop->warp->network; generate from one of later layer-> feed into SVM

（=> Use b-box of MCG region instead of Selective Search b-box; Use penultimate fully connected layer)

#### feature extractors

- A: 
 - finetuned for detection to extract feature of MCG bounding box
 (contain no information about actual region foreground =>)
 - second set of features; different input: with background masked out
 - Concaternating two feature vectores
- B:
 - different network finetuned to extract second set of features 
   change labels: based on segmentation overlap of the region with a GT region (instead of b-box)
- C:
 - train the neworks jointly: disjoint except final classifier layer
   two pathway: box and region pathway
 - A and B are both instantiations of C
 - training: each pathway initialized with network finetuned on each input, and then finetune jointly
 - test: discard final classifier layer, use the output of penultimate

![SDS_net](http://cs.unc.edu/~zhenni/blog/notes/images/SDS_net.png)

### Region classification

- train linear SVM:
 - train initial one: positives: GT; negatives: regions overlapped GT less than 20%
 - retrain with new positive set: highest scoring MCG candidates overlapped the GT more than 50%, discard GTs has no such condidates (work better than initial one positive set)
- Test: 
 - use region classifier to score regions
 - overlapping -> strict non-max suppression with threshold 0 (pixel support should not overlap)
    only use top 20k detection per category (no effect on $$\text{AP}^{r}$$ metric)

### Region refinement

- predict a coarse, top-down figure-ground mask for each region
 - b-box->pad-> discretize into 10×10 grid
 - for each grid cell, train logistic regression classifier to predict prob that the pixel is in foreground
   feature: CNN and figre-ground mask discretized to the same 10×10☆ 
   training set: overlap more than 70% with a GT
- train a second stage to combine this coarse mask with region candidate. ☆
 - "project coarse mask in the superpixels by assigning to each superpixel the average value of coarse mask in the superpixel, classify each superpixel, using as features this projected value in the superpixel and a 0 or 1 encoding if the superpixel belongs to the original region candidate."