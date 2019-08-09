# ATTEND-GAN Model

TensorFlow implementation of [Towards Generating Stylized Image Captions via Adversarial Training]().
<p align="center">
<img src="./examples/samples.jpg" width=1000 high=700>
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

### Content
1. [Model Train Code](./model_train.py)
2. [Model Test Code](./model_test.py)
3. [ATTEND-GAN Generator](lib/generator_WGAN.py)
4. [ATTEND-GAN Discriminator](lib/discriminator_WGAN.py)

### Train
1. Download [Microsoft COCO Dataset](http://cocodataset.org/#download) including neutral image caption data and [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/) including sentiment-bearing image caption data.
2. Reseize the downloded images into [224, 224] and put them in "./images".
3. Preprosses the COCO image caption data and place them in "./data/neutral". You can do this by [prepro.py](https://github.com/yunjey/show-attend-and-tell) and the ResNet-152 network trained on ImageNet, which is generating [7,7,2048] feature map (we use the Res5c layer of the network).
4. Preprosses the SentiCap image caption data and place its positve part in "./data/positive" and its negative part in "./data/negative". You can do this by [prepro.py](https://github.com/yunjey/show-attend-and-tell) and the ResNet-152 network trained on ImageNet, which is generating [7,7,2048] feature map (we use the Res5c layer of the network).
5. Pretrain the generator and discriminator using "./data/neutral". (python model_train.py)
6. Train the generator and the discriminator using "./data/positive" for the positive part and "./data/negative" for the negative part. (python model_train.py)

### Test
1. Dowload [pretrained models]() and unzip the models in "./models".
2. python model_test.py

### Results
|                   | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE
|-------------------|:-------------------:|:------------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
|ATTEND-GAN| 56.55%  | 13.05% | 18.35%  | 44.45%  | 62.85% | 16.05%  |

ATTEND-GAN is inspired from [Self-critical Sequence Training](https://github.com/weili-ict/SelfCriticalSequenceTraining-tensorflow) and [SeqGAN](https://github.com/LantaoYu/SeqGAN) in TensorFlow.
