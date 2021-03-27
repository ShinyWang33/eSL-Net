

# eSL-Net

<div  align="center">    
<img src="figs/show.png" width = "600"  alt="show" align=center />   
</div>

<div  align="">    
Figure 1. Our eSL-Net reconstructs high-resolution, sharp and clear intensity images for event cameras by APS frames and the corresponding event sequences.
</div>



This is code for  the paper **Event Enhanced High-Quality Image Recovery** by Bishan Wang, Jingwei He, Lei Yu, Gui-Song Xia, Wen Yang.

You can find a pdf of the paper [here](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580154.pdf). The paper has been accepted by ECCV2020. If you use of this code or the synthetic dataset, please cite the following publications:

```
@inproceedings{wang2020event,
  title={Event Enhanced High-Quality Image Recovery},
  author={Wang, Bishan and He, Jingwei and Yu, Lei and Xia, Gui-Song and Yang, Wen},
  booktitle={European Conference on Computer Vision},
  year={2020},
  organization={Springer}
}
```

## Run

- Pretrained model with SR : code/pre_trained/model_withsr_pretrained.pt

- Pretrained model without SR : code/pre_trained/model_withoutsr_pretrained.pt

- Model of eSL-Net with SR: code/model_sr.py

- Model of eSL-Net without SR: code/model_withoutsr.py

- Example files with event data: data_example

  - Data of APS frames: data_example/cups/images、 data_example/pic2/images

  - Data of events: data_example/cups/mat、 data_example/pic2/mat

    if you have new event data, you can preprocess the events referring to data_example/cups/mat and data_example/pic2/mat.

  - the path of loading input for eSL-Net: realdata_list.txt

- Run reconstruction:

  ```
  cd code
  ```

  if you want to reconstruct images without SR:

  ```
  python test_realdata.py --sr=0 --model=pre_trained/model_withoutsr_pretrained.pt --num_frame=3 --output_path=realdata_dn/
  ```

  if you want to reconstruct images with SR:

  ```
  python test_realdata.py --sr=1 --model=pre_trained/model_withsr_pretrained.pt --num_frame=3 --output_path=realdata_sr/
  ```

## Synthetic Dataset

This synthetic dataset is generated from high-resolution sharp images of [GoPro dataset](https://seungjunnah.github.io/Datasets/reds.html) and [ESIM](http://rpg.ifi.uzh.ch/esim.html). And the process of generating the synthetic dataset is described in detailed in our paper.

Downloads are available via Baidu Net Disk.

|            Type             |                            Train                             |                          Validation                          |
| :-------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    HR clear sharp images    | [train_sharp_hr](https://pan.baidu.com/s/1sy-60ulFrCK8ltxJw10YYA)(password: 1e2d) | [val_sharp_hr](https://pan.baidu.com/s/1-v19Rz8VQthTM01n6qmjlw)(password: we5s) |
|    LR clear sharp images    | [train_sharp_lr](https://pan.baidu.com/s/1SxhObEv9wjtnbmlEAX-w3g)(password: 5qv6) | [val_sharp_lr](https://pan.baidu.com/s/1sEbNc-4s08Pup-4BRsxQ1g)(password: fqkk) |
|   LR noisy blurry images    | [train_blur_lr](https://pan.baidu.com/s/1ebw1eMRBQ6fwfBTb1gCGNw)(password: qbpb) | [val_blur_lr](https://pan.baidu.com/s/13gdiQafhtDU6kCP91fpjig)(password: ngvv) |
| Event sequences with noises | [train_esim](https://pan.baidu.com/s/1zD0P5AcYznMPtlgeNz0dZQ)(password: 8m73) | [val_esim](https://pan.baidu.com/s/1MfyJJ0cydCLvVBeRII90aQ)(password: gwhk) |



## Contents

* [Introduction](#introduction)
* [Results](#results)
  * [Comparisons of Reconstruction on the synthetic dataset](#synthetic-dataset)
  * [Comparisons of Reconstruction on the real dataset](#real-dataset)
  * [High frame-rate Reconstruction ](#High-frame-rate)



## Introduction

With extremely high temporal resolution, event cameras have a large potential for robotics and computer vision. However, the recovering of high-quality images from event cameras is a very challenge problem, where the following issues should be addressed simultaneously.

* **Low frame-rate and blurry intensity images:** The APS (Active Pixel Sensor) frames are with relatively low frame-rate. And the motion blur is inevitable when recording highly dynamic scenes.

* **High level and mixed noises:** The thermal effects or unstable light environment can produce a huge amount of noisy events. Together with the noises from APS frames, the reconstruction of intensity image would fall into a mixed noises problem.
* **Low spatial-resolution:**  The leading commercial event cameras are typically with very low spatial-resolution. And there is a balance between the spatial-resolution and the latency

In our paper,  we propose an explainable network, an **e**vent-enhanced **S**parse **L**earning **Net**work (**eSL-Net**),  to **recover the high-quality images from event cameras**. Since events depict brightness changes, with the enhanced degeneration model by the events, the clear and sharp high-resolution latent images can be recovered from the noisy, blurry and low-resolution intensity observations. Exploiting the framework of sparse learning, the events and the low-resolution intensity observations can be jointly considered. Furthermore, without additional training process, the proposed eSL-Net can be **easily extended to generate continuous frames with frame-rate as high as the events**.

<div  align="center">    
<img src="figs/eSL-Net.PNG" width = "800"  alt="haha" align=center />   
</div>

<div  align="center">    
Figure 2. Architecture of the proposed eSL-Net.
</div>

## Results of Reconstruction

### Qualitative Comparisons of Reconstruction on the synthetic dataset

<div  align="center">    
<img src="results/syn1.PNG" width = "800"  alt="syn_result1" align=center />   
<img src="results/syn2.PNG" width = "800"  alt="syn_result2" align=center /> 
<img src="results/syn3.PNG" width = "800"  alt="syn_result3" align=center />
</div>

<div  align="center">       
 Qualitative comparison of eSL-Net to EDI, CF and MR with SR method on the synthetic dataset.
</div>

| Methods  | EDI+RCAN 4x | CF+RCAN 4x | MR+RCAN 4x | eSL-Net 4x |
| :------: | :---------: | :--------: | :--------: | :--------: |
| PSNR(dB) |    12.88    |   12.89    |   12.89    | **25.41**  |
|   SSIM   |   0.4647    |   0.4638   |   0.4643   | **0.6727** |

<div  align="center">       
 Quantitative comparison of our outputs to EDI, CF and MR with SR method on the synthetic dataset.
</div>

### Qualitative Comparisons of Reconstruction on the real dataset 

<div  align="center">    
<img src="results/real1.PNG" width = "800"  alt="real_result1" align=center />   
<img src="results/real2.PNG" width = "800"  alt="real_result2" align=center /> 
<img src="results/real3.PNG" width = "800"  alt="real_result3" align=center />
</div>

<div  align="center">       
 Qualitative comparison of eSL-Net to EDI, CF and MR with SR method on the real dataset.
</div>

| real data/BRISQUE | EDI+RCAN 4x | CF+RCAN 4x | MR+RCAN 4x | eSL-Net 4x  |
| :---------------: | :---------: | :--------: | :--------: | :---------: |
|   camerashake1    |   55.8542   |  109.122   |  83.9851   | **55.6984** |
|    indoordrop     |   64.1578   |  65.8033   |  80.7871   | **62.5109** |

<div  align="center">       
 Quantitative comparison of eSL-Net to EDI, CF and MR with SR method on the real dataset by BRISQUE measure, where lower values indicate higher quality.
</div>

### High frame-rate Reconstruction

In the following videos, The left side is the original APS frame by bicubic upsampling for 4 times, and the right side are the high frame rate, high resolution reconstructed results of eSL-Net.

**Event Camera——DAVIS240：**

<div align="center">
   <img src="results/camerashake1.gif" width = "500" alt="camerashake" align=center /> 
</div>

<div align="center">
   <img src="results/rotatevideonew2_6.gif" width = "500" alt="rotatevideo" align=center /> </div>

**Event Camera——DAVIS346：**

<div align="center">
    <img src="results/j4.gif" width = "500" alt="j4" align=center /> 
</div>

<div align="center">
    <img src="results/e4.gif" width = "500" alt="e4" align=center /> 
</div>