# CHCANet



[**CHCANet: Two-view Correspondence Pruning with Consensus-guided Hierarchical Context Aggregation**](https://www.sciencedirect.com/science/article/pii/S0031320324010331)

**Gang Wang, Yufei Chen, Bin Wu**

```tex
@article{wang2024chcanet,
  title={CHCANet: Two-view correspondence pruning with Consensus-guided Hierarchical Context Aggregation},
  author={Wang, Gang and Chen, Yufei and Wu, Bin},
  journal={Pattern Recognition},
  pages={111282},
  year={2024},
  publisher={Elsevier}
}	
```

## Requirements

Please use Python 3.7 and Pytorch 1.13. 

Other dependencies should be easily installed through pip or conda.

```bash
pip install -r core/requirements.txt
```

## Train

**Train model on outdoor (yfcc100m) scene**

```bash
bash train.sh
```

or

```bash
python main.py --data_tr=../../data_dump/yfcc-sift-2000-train.hdf5 --data_va=../../data_dump/yfcc-sift-2000-val.hdf5  --log_base=../model/logCHCANet/yfcc_sift --gpu_id=0 
```

**Train model on indoor (sun3d) scene**

```bash
python main.py --data_tr=../../data_dump/sun3d-sift-2000-train.hdf5 --data_va=../../data_dump/sun3d-sift-2000-val.hdf5  --log_base=../model/logCHCANet/sun3d_sift --gpu_id=0 
```

## Test

**Test pretrained model on outdoor (yfcc100m) scene**

```bash
bash test.sh
```

or

```bash
python main.py --run_mode=test --data_te=../../data_dump/yfcc-sift-2000-test.hdf5  --model_path=../model/logCHCANet/yfcc_sift/train/ --res_path=../model/logCHCANet/yfcc_sift/test/ --gpu_id=1 
```

**Test pretrained models on indoor (sun3d) scene**

```bash
python main.py --run_mode=test --data_te=../../data_dump/sun3d-sift-2000-test.hdf5  --model_path=../model/logCHCANet/sun3d_sift/train/ --res_path=../model/logCHCANet/sun3d_sift/test/ --gpu_id=1 
```

## Acknowledgement

This code is heavily borrowed from [zjhthu/OANet](https://github.com/zjhthu/OANet). If you use the part of code related to data generation, testing and evaluation, you should cite this paper and follow its license.

```tex
@article{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  journal={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
