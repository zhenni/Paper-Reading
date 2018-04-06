# Pytorch

tags: Pytorch

## How to finetune a network

1. Modify dataLoader / Class for custom dataset
  1. segmentation tasks
    1. modify the `palette` for rendering the result labels
  2. modify `num_classes`
  3. dataloader part: modify `__init__` and `__getitem__` 
2. Modify training code:
  1. Change dataset.
    - import new dataset functions and classes
    - change initialization for the dataset
    - change mean-std
    - modify all parts associated with dataset: eg. `xxxDataset.num_class`
  2. Net Surgery
    - Load initial net
    - modify 
      - `net_init.final._modules['4'] = nn.Conv2d(512, UAVDataset.num_classes, kernel_size=1).cuda()`
3. s

