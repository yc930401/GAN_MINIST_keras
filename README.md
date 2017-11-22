# GAN_MINIST_keras
Generate MINST digits using generative adversary networks

## Introduction

I have tried many discriminative models previously using CNN. And this time, I tried aa generrative model, called Generative Adversary Networks. This is a very interesting network, oh, two networks actually. One is used to generate fake images that looks like the reall images, the other is used to distinguish the two kind of images. The generator proformance improve while training until the discriminator cannot distinguish between fake images and real ones. </br>
This is how Ian J. GoodFellow wrote in his paper: </br>
> In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.

## Methodology

1. Define a discriminative network
2. Define a genrative network
3. Generate real and fake images
3. Train a discriminative network
4. Fix the discriminative network and train the generative network using discriminative network and continue these two training processes
5. Plot loss, acc and fake and real images

## Result

I trained the network 14 hours on my own PC. It is getting better and better, but more time is needed to train a better generator. I post it here first. </br>
#### Real images
![GAN](/code/Real images.png) </br>
#### Fake images
![GAN](/code/Fake images.png) </br>

## References
https://www.wouterbulten.nl/blog/tech/getting-started-with-generative-adversarial-networks/
https://arxiv.org/abs/1406.2661
https://arxiv.org/abs/1511.06434
