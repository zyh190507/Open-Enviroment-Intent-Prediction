## Source code for 'Towards Open Environment Intent Prediction'.

## Dependencies
### Use anaconda to create python environemnt:
```conda create --name env python=3.7```
### Install all required libraries:
```pip install -r requirements.txt```

## Usage

### Run the experiments (for example T5_prefix_tuning_with_multi_label_v2):
```bash
1. python prefix_trainer.py --dataset banking --lr 2e-4 --pre_seq_len 256 --seed 42 --gpu_id 0
2. python adb_model.py --dataset banking --lr 2e-4 --pre_seq_len 256 --seed 42 --gpu_id 0
3. python generate_ood_class.py --dataset banking --lr 2e-4 --pre_seq_len 256 --seed 42 --p_node 0.2 --gpu_id 0
```

## Results:
Results stored in outputs, e.g. banking_adb_result.csv

## Acknowledgments

+ [Adaptive-Decision-Boundary](https://github.com/hanleizhang/Adaptive-Decision-Boundary) 
+ [PrefixTuning](https://github.com/XiangLi1999/PrefixTuning)
+ [RAMA](https://github.com/pawelswoboda/RAMA)
