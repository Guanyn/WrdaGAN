# WrdaGAN

## Environment
- Python 3.8
- Pytorch 1.11.0

### Environmental installation
```shell script
pip install -r requirements.txt
```
## Dataset
- Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to data/
- Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to data/birds/
- Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to data/coco/images/

## Train
```shell script
python main.py --config ./config/bird.yml --gpus 0 --batch_size 16 --epochs 1001 --output ./output/bird
```

### Resume training process
If your training process is interrupted unexpectedly, set resume_epoch and resume_model_path in main.py to resume training.


## Val
### Tensorboard
Our code supports automate FID evaluation during training, the results are stored in TensorBoard files under `output_dir/logs`. You can change the test interval by changing test_interval in the YAML file.
 
 ```shell script
 tensorboard --logdir=./output/bird/logs --port 8166
```
### Sampling
We support saving sample results for each model tested. You can find it in `output_dir/imgs`. `captions.txt` is the captions of the sample, and `z.png` is the corresponding label.

### Gradio
Gradio based test code is provided. You can see the result directly
```shell script
python gradio_result.py --config ./config/coco.yml --config2 ./config/bird.yml --load_dir ./output
```
