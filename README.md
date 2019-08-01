# ATTEND-GAN Model

TensorFlow implementation of [Towards Generating Stylized Image Captions via Adversarial Training]().
<p align="center">
<img src="./images/samples.jpg" width=1000 high=700>
</p>

### Reference
if you use our codes or models, please cite our paper:
```
@article{,
  title={Towards Generating Stylized Image Captions via Adversarial Training},
  author={Mohamad Nezami, Omid and Dras, Mark and Wan, Stephen and Paris, Cecile and Hamey, Len},
  journal={arXiv preprint},
  year={2019}
}
```
### Data
We pretrain our models using [Microsoft COCO Dataset](http://cocodataset.org/#download). 
Then, we train the models using [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/).

## Requiremens
1. Python 2.7.12
2. Numpy 1.15.2
3. Hickle
4. Python-skimage
3. Tensorflow 1.8.0

### Contents
1. [Train code](./train.py)
2. [Test code](./test.py)
3. [ATTEND-GAN model](./core/model_WGAN.py)

ATTEND-GAN is inspired from [SeqGAN](https://github.com/LantaoYu/SeqGAN) in TensorFlow.
