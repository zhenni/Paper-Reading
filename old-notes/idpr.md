# Articulated Pose Estimation by a Graphical Model with Image Dependent Pairwise Relations

tags: Deep Learning, Computer Vision, Pose Estimation, NIPS 2014project homepage: [http://www.stat.ucla.edu/~xianjie.chen/projects/pose_estimation/pose_estimation.html](http://www.stat.ucla.edu/~xianjie.chen/projects/pose_estimation/pose_estimation.html)<!-- toc -->


## Paper

### Graphical Model
- $$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$
- $$\mathcal{V}$$ : the point of the joints(parts).
- $$\mathcal{E}$$ : spatial relation between the joints(parts).
- $$K = |\mathcal{V}|$$, simply regard as $$K$$-node tree.
- $$l_i = (x, y)$$ - locations : pixel location of each part $$i$$.
- $$t_{ij} \in \{1, 2, ..., T_{ij}\}$$ - types : a mixture of different spatial relationships. 
  $$t = \{t_{ij}, t_{ji} | (i, j) \in \mathcal{E}\}$$ : set of spatial relations 

####Score Function####
- Unary Term
 $$U(l_i, I)$$
- IDPR term 
 $$R(l_i, l_j, t_{ij}, t_{ji} | I)$$

- Full score function: 
$$F(l, t| I) = \sum_{i \in  \mathcal{V}}U(l_i, I) + \sum_{(i, j) \in \mathcal{E}} R(l_i, l_j, t_{ij}, t_{ji} | I) + w_0$$

- - - 
## Implementation
- - - 
### demo.m
`conf` is a structure of the given global configuration. `conf.pa` is the index of the parent of each joint. `p_no` is the number of the parts(joints).
The main part of this function is shown in the following.
```matlab
// read data 
[pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = LSP_data();
// train dcnn
train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file);
// train graphical model
model = train_model(note, pos_val, neg_val, tsize);
// testing
boxes = test_model([note,'_LSP'], model, pos_test);
/* ... */
// evaluation
show_eval(pos_test, ests, conf, eval_method);
```

- - -

### Read data : `LSP_data.m`
Some variables and constants:
```matlab
trainval_frs_pos = 1:1000;      // training frames for positive
test_frs_pos = 1001:2000;       // testing  frames for positive
trainval_frs_neg = 615:1832;    // training frames for negative (of size 1218)
frs_pos = cat(2, trainval_frs_pos, test_frs_pos); // frames for negative
all_pos                         // num(pos)*1 struct array for positive
                                // struct: im, joints, r_degree, isflip
neg                             // num(neg)*1 struct array for negative
pos_trainval = all_pos(1 : numel(trainval_frs_pos));  // training and validation image struct for positive
pos_test = all_pos(numel(trainval_frs_pos)+1 : end);  // testing image struct for positive
```

Data preparing:

- `lsp_pc2oc` : `function joints = lsp_pc2oc(joints)` : convert to person-centric
- `pos_trainval(ii).joints = Trans * pos_trainval(ii).joints;` Create ground truth joints for model training. Augment the original 14 joint positions with midpoints of joints, defining a total of 26 joints
- `add_flip` : flip trainval images (horizontally) (\#pos\_trainval *= 2)
- `init_scale` : init dataset specific parameters
- `add_rotate` : rotate trainval images (every $9^{\circ}$) (\#pos\_trainval *= 40)
- `val_id = randperm(numel(pos_trainval), 2000);` : split training and validation data for positive (random choose 2000 image from the `pos_trainval` to be the validation set, \#training  = \#pos\_trianval - 2000 = 78000)

- `val_id = randperm(numel(neg), 500);` split training and validation data for negtive (random choose 500 image from the `neg` to be the validation set, \#neg\_val  = \#neg - \#neg\_train = 1218 - 500 = 728)
- `add_flip` : flip the negative data (\#neg\_val *= 2; \#neg\_train *= 2)

 - - -
 
### Train DCNN : `train_dcnn.m`

Some variable and constants:
```
mean_pixel = [128, 128, 128];           // the mean value of each pixel
K = conf.K;                             // K = T_{ij}
```
#### Prepare patches : `prepare_patches.m`
Prepare the patches and derive their labels to train dcnn

##### K-means : get $$r_{ij}$$, $$t_{ij} $$ and the labels $$\cup_{c = 0}^{K}\{c\}\times (\times_{j \in \mathbb{N}(i)} \{1, 2, ..., T_{ij}\})$$

```matlab
// generate the labels
clusters = learn_clusters(pos_train, pos_val, tsize);
label_train = derive_labels('train', clusters, pos_train, tsize);
label_val = derive_labels('val', clusters, pos_val, tsize);

// labels for negative (dummy)
dummy_label = struct('mix_id', cell(numel(neg_train), 1), ...
    'global_id', cell(numel(neg_train), 1));

// all the training data
train_imdata = cat(1, num2cell(pos_train), num2cell(neg_train));
train_labels = cat(1, num2cell(label_train), num2cell(dummy_label));

// random permute the data and store it in the format of LMDB
perm_idx = randperm(numel(train_imdata));
train_imdata = train_imdata(perm_idx);
train_labels = train_labels(perm_idx);
if ~exist([cachedir, 'LMDB_train'], 'dir')
    store_patch(train_imdata, train_labels, psize, [cachedir, 'LMDB_train']);
end
// validation data for positive
val_imdata = num2cell(pos_val);
val_labels = num2cell(label_val);
if ~exist([cachedir, 'LMDB_val'], 'dir')
    store_patch(val_imdata, val_labels, psize, [cachedir, 'LMDB_val']);
end
```
###### Learn clusters : `learn_clusters`(call `cluster_rp` cluster relative position) 

- `nbh_IDs = get_IDs(pa, K);`: get the neighbor of each part(joint)
- `clusters{ii}`: cell : the mean relative postion of `ii`-th part
- k-means
 - `X(ii,:) = norm_rp(imdata(ii), cur,  nbh, tsize);`  relative position for `ii`-th data item
 - `mean_X = mean(X(valid_idx,:),1);` 
 `normX = bsxfun(@minus, X(valid_idx,:), mean_X);` centralize (normalize) the relative position
 - Run `R` trials of the k-means algorithm and choose the one has the smallest distance
 `[gInd{trial}, cen{trial}, sumdist(trial)] = k_means(normX, K);`
 calculate the `imgid`(all the img belongs to the cluster `k`)  of `clusters{cur}{n}(k)`, where `clusters{cur}{n}(k)` is the `k`-th cluster of `n`-th neighbor of the `cur`-th joint.

###### Derive labels


- - - 
#### Train dcnn
System call `caffe` to train dcnn
```matlab
system([caffe_root, '/build/tools/caffe train ', sprintf('-gpu %d -solver %s', ...
    conf.device_id, caffe_solver_file)]);
```

##### Get fully-convolutional net : `net_surgery.m`
