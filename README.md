# DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection

PyTorch implementation of DASVDD.

<center><img src="https://github.com/Armanfard-Lab/DASVDD/blob/main/Figs/Overview.png" alt="Overview" width="800" align="center"></center>

## Citation

You can find the preprint of our paper on [arXiv](https://arxiv.org/abs/2106.05410).

Please cite our paper if you use the results of our work.

```
@inproceedings{Hojjati2021DASVDDDA,
  title={DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection},
  author={H. Hojjati and N. Armanfard},
  year={2021}
}
```

## Abstract

>Semi-supervised anomaly detection, which aims to detect anomalies from normal samples using a model that is solely trained on normal data, has been an active field of research in the past decade. With recent advancements in deep learning, particularly generative adversarial networks and autoencoders, researchers have designed efficient deep anomaly detection methods. Existing works commonly use neural networks such as an autoencoder to map the data into a new representation that is easier to work with and then apply an anomaly detection algorithm. In this paper, we propose a method, DASVDD, that jointly learns the parameters of an autoencoder while minimizing the volume of an enclosing hyper-sphere on its latent representation. We propose a customized anomaly score which is a combination of autoencoder's reconstruction error and distance of the lower-dimensional representation of a sample from the center of the enclosing hyper-sphere. Minimizing this anomaly score on the normal data during training aids us in learning the underlying distribution of normal data. Including the reconstruction error in the anomaly score ensures that DASVDD does not suffer from the common hyper-sphere collapse issue since the proposed DASVDD model does not converge to the trivial solution of mapping all inputs to a constant point in the latent representation. Experimental evaluations on several benchmark datasets from different domains show that the proposed method outperforms most of the commonly used state-of-the-art anomaly detection algorithms while maintaining robust and accurate performance across different anomaly classes.

## Example Anomalies

<img src="https://github.com/Armanfard-Lab/DASVDD/blob/main/Figs/anomaly.png" alt="Results" width="800">


