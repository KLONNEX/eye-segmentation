# eye-segmentation
This repository was forked from mmsegmentation, but by copying files, because no able to set private 
if repo was forked.


## Usage
### Training

```bash
bash tools/dist_train.sh configs/swin/swin_config.py [GPU_NUM]
```
- GPU_NUM - количество гпу для распределенного обучения.

### Inference

```bash
python tools/test.py configs/swin/swin_config.py [CHECKPOINT_PATH] --show_dir [OUTPUT_PATH] --opacity 1
```
- CHECKPOINT_PATH - путь до обученной модели (чекпоинт).
- OUTPUT_PATH - путь до папки, в которую будут сохраняться результаты.