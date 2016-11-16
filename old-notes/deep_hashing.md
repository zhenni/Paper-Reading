# Deep Hashing

Hashing BaseLine: https://github.com/willard-yuan/hashing-baseline-for-image-retrieval

Some representative papers about deep hashing

## Recent Papers
1. [x] **(CNNH) Supervised Hashing via Image Representation Learning** [[paper](http://ss.sysu.edu.cn/%7Epy/papers/AAAI-CNNH.pdf)][[code](http://ss.sysu.edu.cn/%7Epy/CNNH/cnnh.html)][[slide](http://ss.sysu.edu.cn/%7Epy/CNNH-slides.pdf)]  
  Rongkai Xia, Yan Pan, Hanjiang Lai, Cong Liu, and Shuicheng Yan. [AAAI], 2014 
- [ ] **(NINH) Simultaneous Feature Learning and Hash Coding with Deep Neural Networks** [[paper](http://arxiv.org/pdf/1504.03410v1.pdf)]  
  Hanjiang Lai, Yan Pan, Ye Liu, and Shuicheng Yan. [CVPR], 2015
- [ ] **(DRSDH) Bit-Scalable Deep Hashing With Regularized Similarity Learning for Image Retrieval and Person Re-Identification** [[paper](http://arxiv.org/pdf/1508.04535v2.pdf)][[code](https://github.com/ruixuejianfei/BitScalableDeepHash)]  
  Ruimao Zhang, Liang Lin, Rui Zhang, Wangmeng Zuo, and Lei Zhang. [TIP], 2015
- [ ] **Convolutional Neural Networks for Text Hashing** [[paper](http://ijcai.org/papers15/Papers/IJCAI15-197.pdf)]  
  Jiaming Xu, PengWang, Guanhua Tian, Bo Xu, Jun Zhao, Fangyuan Wang, Hongwei Hao. [IJCAI], 2015
- [x] **(DSRH) Deep Semantic Ranking Based Hashing for Multi-Label Image Retrieval** [[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Deep_Semantic_Ranking_2015_CVPR_paper.pdf)][[code](https://github.com/zhaofang0627/cuda-convnet-for-hashing)]  
  Fang Zhao, Yongzhen Huang, Liang Wang, and Tieniu Tan. [CVPR], 2015  
- [x] **(DH) Deep Hashing for Compact Binary Codes Learning** [[paper](https://sites.google.com/site/elujiwen/CVPR15b.pdf?attredirects=0&d=1)]  
   Venice Erin Liong, Jiwen Lu, Gang Wang, Pierre Moulin, and Jie Zhou. [CVPR], 2015
- [x] **Deep Learning of Binary Hash Codes for Fast Image Retrieval** [[paper](http://www.iis.sinica.edu.tw/%7Ekevinlin311.tw/cvprw15.pdf)][[code](https://github.com/kevinlin311tw/caffe-cvprw15)][[questions](http://www.iis.sinica.edu.tw/%7Ekevinlin311.tw/deephash_questions.txt)]  
  Kevin Lin, Huei-Fang Yang, Jen-Hao Hsiao, and Chu-Song Chen. [CVPRW], 2015
- [x] **(DPSH) Feature Learning based Deep Supervised Hashing with Pairwise Labels** [[paper](http://arxiv.org/pdf/1511.03855v1.pdf)][[code](http://cs.nju.edu.cn/lwj/code/DPSH_code.rar)]  
- [ ] **Deep Learning to Hash with Multiple Representations** [[paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6413830)]   
Yoonseop Kang, Saehoon Kim, Seungjin Choi. [ACMMM], 2012
- [ ] **Inductive Transfer Deep Hashing for Image Retrieval**[[paper](http://delivery.acm.org/10.1145/2660000/2654987/p969-ou.pdf?ip=59.78.61.68&id=2654987&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E17676C47DFB149BF%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=592513902&CFTOKEN=30073573&__acm__=1458232968_d23bec2e64b324c578951cef7c4f98af)]  
  Xinyu Ou, Lingyu Yan, Hefei Ling∗ , Cong Liu, Maolin Liu
- [ ] **A Deep Hashing Learning Network** [[paper](http://arxiv.org/pdf/1507.04437v1.pdf)]

## Details

- **(CNNH) Supervised Hashing via Image Representation Learning** [[paper](http://ss.sysu.edu.cn/%7Epy/papers/AAAI-CNNH.pdf)][[code](http://ss.sysu.edu.cn/%7Epy/CNNH/cnnh.html)][[slide](http://ss.sysu.edu.cn/%7Epy/CNNH-slides.pdf)]  
  Rongkai Xia, Yan Pan, Hanjiang Lai, Cong Liu, and Shuicheng Yan. [AAAI], 2014 
>1.Given the pairwise similarity matrix $$S$$ over training images, they use a scalable coordinate descent method to decompose $$S$$ into a product of $$HH^T$$ where $$H$$ is a matrix with each of its rows being the approximate hash code associated to a training image.

 <img src="http://cs.unc.edu/~zhenni/blog/notes/images/CNNH-stage1.png" width = "400" alt="CNNH-stage1" align=center />
>2.In the second stage, the idea is to simultaneously learn a good feature representation for the input images as well as a set of hash functions, via a deep convolutional network tailored to the learned hash codes in $$H$$ and optionally the discrete class labels of the images. (Using Alexnet)
 
 ![CNNH](http://cs.unc.edu/~zhenni/blog/notes/images/CNNH.png)

- **(NINH)Simultaneous Feature Learning and Hash Coding with Deep Neural Networks** [[paper](http://arxiv.org/pdf/1504.03410v1.pdf)]  
  Hanjiang Lai, Yan Pan, Ye Liu, and Shuicheng Yan. [CVPR], 2015
  >The pipeline of the proposed deep architecture consists of three building blocks: 1) a sub-network with a stack of convolution layers to produce the effective intermediate image features; 2) a divide-and-encode module to divide the intermediate image features into multiple branches, each encoded into one hash bit; and 3) a triplet ranking loss designed to characterize that one image is more similar to the second image than to the third one.
  
  ![SFH](http://cs.unc.edu/~zhenni/blog/notes/images/SFH.png)
  
- **(DRSDH) Bit-Scalable Deep Hashing With Regularized Similarity Learning for Image Retrieval and Person Re-Identification** [[paper](http://arxiv.org/pdf/1508.04535v2.pdf)][[code](https://github.com/ruixuejianfei/BitScalableDeepHash)]  
  Ruimao Zhang, Liang Lin, Rui Zhang, Wangmeng Zuo, and Lei Zhang. [TIP], 2015
  >We pose hashing learning as a problem of regularized similarity learning. Specifically, we organize the training images into a batch of triplet samples, each sample containing two images with the same label and one with a different label. With these triplet samples, we maximize the margin between matched pairs and mismatched pairs in the Hamming space. In addition, a regularization term is introduced to enforce the adjacency consistency, i.e., images of similar appearances should have similar codes. The deep convolutional neural network is utilized to train the model in an end-to-end fashion, where discriminative image features and hash functions are simultaneously optimized.
  
 <img src="http://cs.unc.edu/~zhenni/blog/notes/images/triplets.png" width = "400" alt="triplets" align=center />
 <img src="http://cs.unc.edu/~zhenni/blog/notes/images/DRSCH.png" width = "400" alt="DRSCH" align=center />
  
- **Convolutional Neural Networks for Text Hashing** [[paper](http://ijcai.org/papers15/Papers/IJCAI15-197.pdf) not found]  
  Jiaming Xu, PengWang, Guanhua Tian, Bo Xu, Jun Zhao, Fangyuan Wang, Hongwei Hao. [IJCAI], 2015
- **(DSRH) Deep Semantic Ranking Based Hashing for Multi-Label Image Retrieval** [[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Deep_Semantic_Ranking_2015_CVPR_paper.pdf)][[code](https://github.com/zhaofang0627/cuda-convnet-for-hashing)]  
  Fang Zhao, Yongzhen Huang, Liang Wang, and Tieniu Tan. [CVPR], 2015
>Deep convolutional neural network is incorporated into hash functions to jointly learn feature representations and mappings from them to hash codes, which avoids the limitation of semantic representation power of hand-crafted features. Meanwhile, a ranking list that encodes the multilevel similarity information is employed to guide the learning of such deep hash functions. An effective scheme based on surrogate loss is used to solve the intractable optimization problem of non-smooth and multivariate ranking measures involved in the learning procedure.  

 1. Deep Hash Functions: $$\displaystyle h(x; w) = sign(w^T[f_a(x); f_b(x)])$$  
 2. Semantic Ranking Supervision: (preserve multilevel semantic structure)various evaluation criteria can be used to measure the consistency of the rankings predicted by hash functions, such as the Normalized Discounted Cu- mulative Gain (NDCG) score: $$\displaystyle NDCG@p=\frac{1}{Z}\sum\limits_{i=1}^{p}\frac{2^{r_i}-1}{\log(1+i)}$$, where $$p$$ is the truncated position in a ranking list, $$Z$$ is a normalization constant to ensure that the NDCG score for the correct ranking is one, and $$r_i$$ is the similarity level of the $$i$$-th database point in the ranking list.
 3. Optimization with Surrogate Loss:  
 Given a query $$q$$ and a ranking list $$\{x_i\}^M_{i=1}$$ for $$q$$, we can define a ranking loss on a set of triplets of hash codes as follows: $$\displaystyle L_\omega(h(q),\{h(x_i)\}^M_{i=1})=\sum\limits_{i=1}^M\sum\limits_{j:r_j<r_i}\omega(r_i, r_j)[d_H(h(q), h(x_i)) - d_H(h(q), h(x_j))+\rho]_+$$. According to NDCD, weight $$\omega(r_i, r_j)= \frac{2^{r_i}-2^{r_j}}{Z}$$    
 The objective function can be given by the empirical loss subject to some regularization: 

$$
\mathcal{F}(W)=\sum\limits_{q\in\mathcal{D}, \{x_i\}_{i=1}^M\subset\mathcal{D}}L_{\omega} ( h(q;W) , \{h(x_i;W) \} ^M_{i=1}) +  \frac{\alpha}{2} || mean_{q}(h(q;W))||_2^2 + \frac{\beta}{2}||W||_2^2
$$

  
 And calculate derivative values.
 
 <img src="http://cs.unc.edu/~zhenni/blog/notes/images/DSRH.png" width = "400" alt="DSRH" align=center />
 <img src="http://cs.unc.edu/~zhenni/blog/notes/images/DSRH-2.png" width = "400" alt="DSRH-2" align=center />
 
- **(DH) Deep Hashing for Compact Binary Codes Learning** [[paper](https://sites.google.com/site/elujiwen/CVPR15b.pdf?attredirects=0&d=1)]  
  Venice Erin Liong, Jiwen Lu, Gang Wang, Pierre Moulin, and Jie Zhou. [CVPR], 2015
  >Our model is learned under three constraints at the top layer of the deep network: 1) the loss between the original real-valued feature descriptor and the learned binary vector is minimized, 2) the binary codes distribute evenly on each bit, and 3) different bits are as independent as possible. To further improve the discriminative power of the learned binary codes, we extend DH into supervised DH (SDH) by including one discriminative term into the objective function of DH which simultaneously maximizes the inter-class variations and minimizes the intra-class variations of the learned binary codes.
  
  1. DH Loss function: $$J = J_1 -\lambda _1J_2 +\lambda_ 2J_3 +\lambda _3J_4$$, where $$J_1 = \frac{1}{2}||B-H^M||_F^2$$ is the quantization loss, $$J_2= \frac{1}{2N} tr(H^M(H^M)^T)$$ is the balance bits constraint, $$J_3 = \frac{1}{2}\sum\limits_{m=1}^M||W^m(W^m)T-I||_F^2$$ is the independent bit constraint, and $$J_4 = \frac{1}{2}(||W^m||^2_F+||c^m||_2^2)$$ are regularizers to control scales of parameters.
  2. SDH (Supervised): $$J_2 = \frac{1}{2}(tr(\frac{1}{N}H^M(H^M)^T)+ \alpha tr(\Sigma_B-\Sigma_W))$$,
  where $$\displaystyle\Sigma_W=\frac{1}{N_S}\sum\limits_{(x_i, x_j)\in\mathcal{S}}(h_i^M-h_j^M)(h_i^M-h_j^M)^T$$, $$\displaystyle\Sigma_B=\frac{1}{N_D}\sum\limits_{(x_i, x_j)\in\mathcal{D}}(h_i^M-h_j^M)(h_i^M-h_j^M)^T$$, and two sets $$\mathcal{S}$$  or $$\mathcal{D}$$ from the training set, which represents the positive samples pairs and the negative samples pairs in the training set, respectively.
  <img src="http://cs.unc.edu/~zhenni/blog/notes/images/DH.png" width = "400" alt="DH" align=center />
  <img src="http://cs.unc.edu/~zhenni/blog/notes/images/DH-alg.png" width = "400" alt="DH-alg" align=center />
  <img src="http://cs.unc.edu/~zhenni/blog/notes/images/SDH-alg.png" width = "400" alt="SDH-alg" align=center />  
- **Deep Learning of Binary Hash Codes for Fast Image Retrieval** [[paper](http://www.iis.sinica.edu.tw/%7Ekevinlin311.tw/cvprw15.pdf)][[code](https://github.com/kevinlin311tw/caffe-cvprw15)][[questions](http://www.iis.sinica.edu.tw/%7Ekevinlin311.tw/deephash_questions.txt)]  
  Kevin Lin, Huei-Fang Yang, Jen-Hao Hsiao, and Chu-Song Chen. [CVPRW], 2015
  1. Learning Hash-like Binary Codes: Add a latent layer $$H$$  between $$F_7$$ and $$F_8$$ to represent the hash code layer. The neurons in the latent layer H are activated by sigmoid functions.The initial random weights of latent layer $H$ acts like LSH.
  2. Coarse-level Search: The binary codes are then obtained by binarizing the activations by a threshold. (1, if $$\geq 0.5$$, 0, $$o.w.$$). Then we can get a pool of candidates.
  3. Fine-level Search: Use the euclidean distances of $$F_7$$ layer feature.
  ![img](http://cs.unc.edu/~zhenni/blog/notes/images/Kevin.png)
- **(DPSH) Feature Learning based Deep Supervised Hashing with Pairwise Labels** [[paper](http://arxiv.org/pdf/1511.03855v1.pdf)][[code](http://cs.nju.edu.cn/lwj/code/DPSH_code.rar)]  
  Wu-Jun Li, Sheng Wang and Wang-Cheng Kang. [arXiv], 2015
  
  1. Define the pairwise loss function similar to that of LFH: $$\displaystyle L = -\log p(\mathcal{S}|\mathcal{B}) = - \log p(\mathcal{s}_{ij}|\mathcal{B}) = -\sum\limits_{\mathcal{s}_{ij}\in\mathcal{S}} (\mathcal{s}_{ij}\theta_{ij}-\log(1 + e^{\theta_{ij}} ))$$, where $$\mathcal{B}=\{b_i\}^n_{i=1},\theta_{ij}=\frac{1}{2}b^T_ib_j$$
  2. Compute the derivatives of the loss function with respect to the (relaxed) hash codes as follows: $$\displaystyle \frac{\partial L}{\partial u_i} = \frac{1}{2}\sum\limits_{j:s_{ij}\in\mathcal{S}}(a_{ij}-s{ij})u_j + \frac{1}{2}\sum\limits_{j:s_{ji}\in\mathcal{S}}(a_{ji}-s{ji})u_j$$, where $$a_{ij} = \sigma(\frac{1}{2}u_i^Tu_j)$$ with $$\sigma(x)=\frac{1}{1+e^{-x}}$$
  
  ![DPSH](http://cs.unc.edu/~zhenni/blog/notes/images/DPSH.png)
  
