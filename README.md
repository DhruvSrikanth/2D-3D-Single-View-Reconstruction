<H1> Resource-Conscious High-Performance Models for 2D-to-3D Single View Reconstruction
  
---
  
<br>
  
<H2> Capstone Project, Research Work and Research Paper </H2>
  
---
  
<br>
  
Our paper that has been published based on the approaches established in this repository can be found here - 

[Resource-Conscious High-Performance Models for 2D-to-3D Single View Reconstruction (Link needs to be updated once IEEE releases paper link)](https://arxiv.org/abs/1901.11153)

---
  
<br>

<H3> Aknowledgements </H3>

---
  
<br>
  
We, **Dhruv Srikanth**, **Suraj Bidnur** and **Rishab Kumar** would like to thank **Dr. Sanjeev G** for his guidance throughout our research and capstone project for our final year of undergraduate engineering. We would also like to thank the **IEEE Society** for publishing our paper titled - **"Resource-Conscious High-Performance Models for 2D-to-3D Single View Reconstruction" by Suraj Bidnur, Dhruv Srikanth and Sanjeev G**.

---
  
<br>

<H3> Objective </H3> 

---
  
<br>
  
We aim to reconstruct 3D voxel models from their 2D images using deep learning algorithms. We differentiate from other techniques, methods and models used in our success in reducing resource utilization, increasing computational efficiency and reducing training time all while improving on the performance and accuracy.

<br>
  
---

<H3> Inspiration </H3> 

---
  
<br>
  
The **Pix2Vox model** and **3D-R2N2** architectures provided us with inspiration. We based original based our approach off of a similar model and then made alteration from that point onwards for single view image reconstruction without any data augmentation. The papers for the **Pix2Vox** and **3D-R2N2** architectures can be found below - 
  
1. [Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images](https://arxiv.org/abs/1901.11153)
2. [3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/abs/1604.00449)

<br>
  
---

<H3> Motivation </H3>  

---
  
<br>
  
* Lack of 3D content despite increasing demands by various industries like gaming, medical, cinema etc.
* Increase in popularity along with the proven success of deep learning techniques like CNNs, GANs etc. over recent years.
* High resource requirements and computation costs in existing approaches.

<br>
  
---
  
<H3> Dataset </H3>

---
  
<br>
  
The dataset we have trained our models on is the 3D [ShapeNet](https://shapenet.org) dataset. The links to the 2D rendering files and the 3D binvox files are mentioned below.  

1. ShapeNet Rendering: <http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz>
2. ShapeNet Binvox: <http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz> 

The dataset contains 13 different object classes with over 700,000 images.

<br>
  
---

<H3> Metrics </H3>

---
  
<br>
  
1. Performance Metric - Intersection over Union (IoU)
2. Loss - Binary Cross-Entropy (BCE)

<br>
  
---


<H3> Training Configuration </H3>  

---
  
<br>
  
1. Epochs: 150
2. Learning Rate: 0.001
3. Input shape: 224,224,3
4. Batch size: 32


<br>
  
---

<H3> Hardware Configuration </H3>  

---
  
<br>
  
1. GPU: Nvidia Tesla T4 with 16GB VRAM
2. CPU and RAM: 4 vCPU’s and 28GB RAM
3. OS: Ubuntu 18.04 running in a Microsoft Azure VM

<br>
  
---

<H3> Software Configuration </H3>  

---
  
<br>
  
1. Tensorflow: 2.4.0
2. CUDA: 11.0
3. cuDNN: 8.0
4. Python: 3.6-3.8

<br>
  
---

<H3> Training Results </H3>  

---
  
<br>
  
Given below are the **mean IoUs** for each of the following models that we trained:
  
1. AE-Res: 0.6787
2. AE-Dense: 0.7258
3. 3D-SkipNet: 0.6871
4. 3D-SkipNet with kernel splitting: 0.6626

Given below are the **mean IoUs** for each of the following models that are state-of-the-art baselines for comparison:
  
1. Pix2Vox: 0.6340
2. 3D-R2N2: 0.5600

<br>
  
---

<H3> Research Paper </H3>  

---
  
<br>
  
Our paper that has been published based on the approaches established in this repository can be found here - 

[Resource-Conscious High-Performance Models for 2D-to-3D Single View Reconstruction (Link needs to be updated once IEEE releases paper link)](https://arxiv.org/abs/1901.11153)

<br>
  
---

<H3> Takeaways </H3>  

---
  
<br>
  
* **There exists a trade-off for skip connections and dense connections between performance and resource utilization.**
* **We propose using dense connections in non-resource constrained environments.**
* **We hope that our models establish the potential to utilise 3D reconstruction of objects whilst utilising minimal resources towards building sustainability in the environment and accessibility on the edge.**

<br>
  
---
  
***
