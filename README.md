# DANet
DANet: Image Deraining via Dynamic Association Learning
# DANet: Image Deraining via Dynamic Association Learning (IJCAI 2022)

[Kui Jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl), [Zhongyuan Wang](https://dblp.org/pid/84/6394.html), [Zheng Wang](https://scholar.google.com/citations?user=-WHTbpUAAAAJ&hl=zh-CN), [Peng Yi](https://dblp.org/pid/98/1202.html), [Junjuan Jiang]([https://scholar.google.com/citations?user=TuEwcZ0AAAAJ&hl=zh-CN](https://scholar.google.com/citations?user=WNH2_rgAAAAJ&hl=zh-CN)), [Jinsheng Xiao]([https://dblp.org/pid/49/67.html](https://scholar.google.com/citations?hl=zh-CN&user=_3XTzN8AAAAJ)), and [Chia-Wen Lin](https://scholar.google.com/citations?user=fXN3dl0AAAAJ&hl=zh-CN)

**Paper**: [Rain-free and Residue Hand-in-Hand: A Progressive Coupled Network for Real-Time Image Deraining](https://www.researchgate.net/publication/353620456_Rain-free_and_Residue_Hand-in-Hand_A_Progressive_Coupled_Network_for_Real-Time_Image_Deraining)



> **Abstract:** *Rain streaks and background components in a rainy input are highly correlated, making the deraining task a composition of the rain streak removal and background restoration. However, the correlation of these two components is barely considered, leading to unsatisfied deraining results. To this end, we propose a dynamic associated network (DANet) to achieve the association learning between rain streak removal and background recovery. There are two key aspects to fulfill the association learning: 1) DANet unveils the latent association knowledge between rain streak prediction and background texture recovery, and leverages it as an extra prior via an associated learning module (ALM) to promote the texture recovery. 2) DANet introduces the parametric association constraint for enhancing the compatibility of deraining model with background reconstruction, enabling it to be automatically learned from the training data. Moreover, we observe that the sampled rainy image enjoys the similar distribution to the original one. We thus propose to learn the rain distribution at the sampling space, and exploit super-resolution to reconstruct high-frequency background details for computation and memory reduction. Our proposed DANet achieves the approximate deraining performance to the state-of-the-art MPRNet but only accounts for 52.6% and 23% inference time and computational cost, respectively.* 

## Network Architecture
<table>
  <tr>
    <td> <img src = "img/PCNet.png" width="500"> </td>
    <td> <img src = "img/CRM.png" width="400"> </td>
  </tr>
</table>

## Results
<p align="center">
  <img src="img/result1.png">
</p>
<p align="center">
  <img src="img/result2.png">
</p>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## Quick Test

To test the pre-trained deraining model on your own images, run 
```
python test.py  
```

## Training and Evaluation

Training and Testing codes for deraining are provided in their respective directories.


## Results
Experiments are performed for different image processing tasks including, image deraining, image dehazing and low-light image enhancement.

## Acknowledgement
Code borrows from [MPRNet](https://github.com/swz30/MPRNet) by [Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en). Thanks for sharing !

## Citation
If you use PCNet, please consider citing:

    @article{jiangDANet2022,
        title={DANet: Image Deraining via Dynamic Association Learning},
        author={Kui Jiang and Zhongyuan Wang and Zheng Wang and Peng Yi and Junjun Jiang and Jinsheng Xiao and Chia-Wen Lin},
        journal={IJCAI.}, 
        year={2022}
    }

## Contact
Should you have any question, please contact Kui Jiang (kuijiang@whu.edu.cn)
