# On the Generalization of Neural Networks Trained with SGD: Information-Theoretical Bounds and Implications


## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Bound Verification

To train MLP for the Bound Verification experiment, run this command:

```
python main.py --arch fc1 --epochs 500 --batch-size 60 --learning-rate 0.01 --dataset mnist --width 512 --label-corrupt-prob 0 --early_stop False --fixinit True --seed 1
```

To train AlexNet for the Bound Verification experiment, run this command:

```
python main.py --arch alexnet --epochs 200 --batch-size 60 --learning-rate 0.001 --dataset cifar10 --width 64 --label-corrupt-prob 0 --early_stop False --fixinit True --seed 1
```

## Dynamic Gradient Clipping

To run the Dynamic Gradient Clipping algorithm for MLP, run this command:

```
python main.py --arch fc1 --epochs 500 --batch-size 60 --learning-rate 0.01 --dataset mnist --width 512 --label-corrupt-prob 0.2 --early_stop False --fixinit True  --clip 50 --clip_factor 0.1 --clip_start 15 --stra 0
```

To run the Dynamic Gradient Clipping algorithm for AlexNet, run this command:

```
python main.py --arch alexnet --epochs 200 --batch-size 60 --learning-rate 0.001 --dataset cifar10 --width 64 --label-corrupt-prob 0.2 --early_stop False --fixinit True  --clip 50 --clip_factor 0.1 --clip_start 20 --stra 0
```

## Gaussian Model Perturbation

To run GMP, run this command:

```
python main.py --dataset svhn --model vgg16 --method gmp  --inner_iter 3 --std 0.03 --lam 0.5
```

