# VoViT: Low Latency Graph-based Audio-Visual VoiceSeparation Transformer
[Project Page](https://ipcv.github.io/VoViT/)  
[Arxiv Paper](https://arxiv.org/abs/2203.04099)  

Accepted to ECCV 2022  
Currently only the speech separation model is uploaded.  
Feel free to request the singing voice separation model.  

### Citation

```
@inproceedings{montesinos2022vovit,
    author = {Montesinos, Juan F. and Kadandale, Venkatesh S. and Haro, Gloria},
    title = {VoViT: Low Latency Graph-Based Audio-Visual Voice Separation Transformer},
    year = {2022},
    isbn = {978-3-031-19835-9},
    publisher = {Springer-Verlag},
    doi = {10.1007/978-3-031-19836-6_18},
    booktitle = {Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVII},
    pages = {310–326},
}
```

## Installation

Download the repo and the face landmark extractor library:

```
https://github.com/JuanFMontesinos/VoViT
cd VoViT
git clone https://github.com/cleardusk/3DDFA_V2
cd 3DDFA_V2
sh ./build.sh
cd ..
```
Download the weights (from the release section) and move them to `VoViT\vovit\core\models\weights`.  
### Requirements  
The core computations (the model itself) depends on python, pytorch, einops and torchaudio. To run demos and visualizations many other libraries are required.

In case of incompatibilities due to future updates, the tested commit is:  
`https://github.com/cleardusk/3DDFA_V2/tree/1b6c67601abffc1e9f248b291708aef0e43b55ae`

## Running a demo

Demos are located in the `demo_samples` folder.  
To process `interview.mp4`  
Modify `inference_interview.py`:  
```
device = 'cuda:0'
path = 'demo_samples/interview'
compute_landmarks = True
```
compute_landmarks implies landmarks will be computed on-the-fly (useful for production-ready models). This needs
curated data (videos cropped around the face etcetera...)
compute_landmarks = False uses preprocessed landmarks extracted from `preprocessing_interview.py`.  

```
python preprocessing_interview.py
python inference_interview.py
```


## Latency

|               | Preprocessing |   Inference   |             | Preprocessing + Inference |
|---------------|:-------------:|:-------------:|:-----------:|:-------------------------:|
|               |               | Graph Network | Whole model |                           |
| VoViT-s1      |     17.95     |      4.50     |    52.21    |           82.18           |
| VoViT         |     17.95     |      4.55     |    57.45    |           93.31           |
| VoViT-s1 fp16 |     10.94     |      2.88     |    30.47    |           52.43           |
| VoViT fp16    |     10.94     |      2.86     |    34.18    |           46.14           |  

Latency estimation for the different variants of VoViT. Average of 10 runs, batch size 100. Device: Nvidia RTX 3090. GPU
utilization >98%, memory on demand. Two forward passed done to warm up. Timing corresponds to ms to process 10s of audio

**Note: Pytorch version is no longer supporting complex32 dtype in pytorch 1.11**  

