# Quantization aware training (ternary and binary)

## Description
This project implement a torch module and function as described in the trained-ternary-quantization paper (https://arxiv.org/pdf/1612.01064v1)

## Modification
Note : the computation of Wn and Wp gradient is different, we computing the mean and not only the sum of the quantized gradient so that we can remove explosing gradient

## Result
coming soon

# Utilisation 
```python
import torch
from quant_module import Conv2D_QuantModule, Conv2D_QuantFunction_Binary, Conv2D_QuantFunction_Ternary

layer = Conv2D_QuantModule(ternary=True/False,  # True -> ternary, False -> binary
                           in_channels=...,
                           out_channels=...,
                           kernel_size=...,
                           stride=...,
                           padding=...,
                           threshold=0.5,       # Change the threshold for experiment
                           Wp_init=-1,          # Change Wp_init for experiment
                           Wn_init=-1)          # Change Wn_init for experiment
image = torch.rand(1, 3, 22, 22)
output = layer(output)
```
