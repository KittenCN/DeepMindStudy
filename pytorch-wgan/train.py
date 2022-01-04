import os
str = "python  D:\workstation\GitHub\DeepMindStudy\pytorch-wgan\main.py --model WGAN-GP \
               --is_train True \
               --download True \
               --dataroot D:\workstation\GitHub\DeepMindStudy\data\cifar10\data\ \
               --dataset cifar \
               --generator_iters 40000 \
               --cuda True \
               --batch_size 64"
os.system(str)