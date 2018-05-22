# Budgeted Super Networks
Original implementation of the [Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks](https://arxiv.org/abs/1706.00046)
 
### Installation
`pip install -r requirements.txt`

### Running
`python bsn_main.py`

The available parameters can be seen using `python bsn_main.py -h`
For exemple to run the Budgeted Super Networks on Cifar10 using the 8 layers/128 channels B-CNF architecture:

`python bsn_main.py -arch CNF -layers 8 -channels 128 -dset CIFAR10`

All plotting is done through [Visdom](https://github.com/facebookresearch/visdom). The server can be configured using the `resources/visdom.json` file.

CUDA usage can be enabled using the `-cuda n` flag, where `n` corresponds to the index of the GPU.
```
# To use the first GPU of the machine:
python bsn_main.py -arch CNF -layers 8 -channels 128 -dset CIFAR10 -cuda 0
```