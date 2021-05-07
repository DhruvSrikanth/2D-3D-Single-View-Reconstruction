<H1> 3D-Reconstruction
  
<H2> B.Tech Electronics and Communication Final Year Capstone Project</H2>  
<br>

This project aims to reconstruct 3D voxel models from their 2D images using mahcine learning algorithms. We have based out project on the **Pix2Vox model** architecture. For now only single view image reconstruction is supported and there is no data augmentation included. The link to the the paper is given below.<br>

[Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images](https://arxiv.org/abs/1901.11153)  
<br>

---

<H3>Dataset</H3>
<br>

The dataset we have trained our models on is the 3D [ShapeNet](https://shapenet.org) dataset. The links to the 2D rendering files and the 3D binvox files are mentioned below.  

1. ShapeNet Rendering: <http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz>
2. ShapeNet Binvox: <http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz>  
<br>

---

<H3>Software</H3>  

1. Tensorflow: 2.4.0
2. CUDA: 11.0
3. cuDNN: 8.0
4. Python: 3.6-3.8

We recommend using a system with an powerful NVIDIA GPU if you want to train the model.  
<br>

***