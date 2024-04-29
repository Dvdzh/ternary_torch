import torch 
import torch.functional as F
import torch.nn as nn

class Conv2D_QuantFunction_Binary(torch.autograd.Function):
    """ 
    Personalized autograd function for binary quantization 
    Forward pass: quantize the weights and then compute the convolution
    Backward pass: compute the gradient of the loss with respect of the input, the weights, Wp and Wn
    """
    
    @staticmethod
    def forward(ctx, X, weight, Wp, Wn, threshold, stride, padding):
            
        # weights normalization
        max_abs_weight = torch.max(torch.abs(weight)).cuda()
        normalized_weight = (weight / max_abs_weight).cuda()
        
        # creating the mask keeping information about the sign of the weights
        threshold = threshold.to(normalized_weight.device)
        mask = torch.where(normalized_weight > threshold, 1, normalized_weight)
        mask = torch.where(mask < threshold, -1, mask).cuda()
        
        # quantization of the weights
        quantized_weight = torch.where(mask == 1, Wp, mask)
        quantized_weight = torch.where(mask == -1, Wn, mask).cuda()
        
        # backward variables saving
        ctx.save_for_backward(X, 
                              quantized_weight,
                              Wp.clone(),
                              Wn.clone(),
                              mask,
                              stride,
                              padding)
        
        return F.conv2d(X, quantized_weight, stride=stride.int().item(), padding=padding.int().item())

    @staticmethod
    def backward(ctx, grad_out):
        
        # context variables loading
        X, quantized_weight, Wp, Wn, mask, stride, padding = ctx.saved_tensors
        
        # gradient of the loss with respect to the input and the weights with quantized weights
        grad_input = torch.nn.grad.conv2d_input(X.shape, quantized_weight, grad_out, stride=stride, padding=padding)
        grad_weight = torch.nn.grad.conv2d_weight(X, quantized_weight.shape, grad_out, stride=stride, padding=padding)
        
        # gradient scaling
        grad_weight = torch.where(mask == 1, Wp * grad_weight, grad_weight)
        grad_weight = torch.where(mask == -1, Wn * grad_weight, grad_weight) 
        
        # gradient of Wp and Wn
        grad_Wp = torch.sum(torch.where(mask == 1, grad_weight, torch.zeros_like(grad_weight))).unsqueeze(0)
        grad_Wn = torch.sum(torch.where(mask == -1, grad_weight, torch.zeros_like(grad_weight))).unsqueeze(0)
        
        # Computing the mean of the positive and negative weights
        size_Wp = torch.sum(torch.where(mask == 1, 1, 0))
        size_Wn = torch.sum(torch.where(mask == -1, 1, 0))
        
        grad_Wp = grad_Wp / size_Wp
        grad_Wn = grad_Wn / size_Wn
        
        return grad_input, grad_weight, grad_Wp, grad_Wn, grad_Wn, None, None # None car pas de gradient pour le threshold

class Conv2D_QuantFunction_Ternary(torch.autograd.Function):
    """ 
    Personalized autograd function for ternary quantization
    Forward pass: quantize the weights and then compute the convolution
    Backward pass: compute the gradient of the loss with respect of the input, the weights, Wp and Wn
    """
    @staticmethod
    def forward(ctx, X, weight, Wp, Wn, threshold, stride, padding):
        
        # weights normalization
        max_abs_weight = torch.max(torch.abs(weight))
        normalized_weight = (weight / max_abs_weight)
        
        # computing the mask which keeps information about the sign of the weights
        threshold = threshold.to(normalized_weight.device)
        mask = torch.where((normalized_weight > -threshold) & (normalized_weight <= threshold), 0, normalized_weight)
        mask = torch.where(mask > threshold, 1, mask)
        mask = torch.where(mask < -threshold, -1, mask)
        
        # quantization of the weights
        quantized_weight = torch.where(mask == 0, 0, mask)
        quantized_weight = torch.where(mask == 1, Wp, mask)
        quantized_weight = torch.where(mask == -1, Wn, mask)

        # sauvegarde des variables pour le backward
        ctx.save_for_backward(X, 
                              quantized_weight,
                              Wp.clone(),
                              Wn.clone(),
                              mask,
                              stride,
                              padding)
        return F.conv2d(X, quantized_weight, stride=stride.int().item(), padding=padding.int().item())

    @staticmethod
    def backward(ctx, grad_out):
        
        # récupération des variables sauvegardées
        X, quantized_weight, Wp, Wn, mask, stride, padding = ctx.saved_tensors
        
        # calcul gradient poids quantifiés et outpu
        grad_input = torch.nn.grad.conv2d_input(X.shape, quantized_weight, grad_out, stride=stride, padding=padding)
        grad_weight = torch.nn.grad.conv2d_weight(X, quantized_weight.shape, grad_out, stride=stride, padding=padding)
        
        # scale le gradient
        grad_weight = torch.where(mask == 1, Wp * grad_weight, grad_weight)
        grad_weight = torch.where(mask == -1, Wn * grad_weight, grad_weight) 
        
        # calcul gradient Wp et Wn
        grad_Wp = torch.sum(torch.where(mask == 1, grad_weight, torch.zeros_like(grad_weight))).unsqueeze(0)
        grad_Wn = torch.sum(torch.where(mask == -1, grad_weight, torch.zeros_like(grad_weight))).unsqueeze(0)
        
        size_Wp = torch.sum(torch.where(mask == 1, 1, 0))
        size_Wn = torch.sum(torch.where(mask == -1, 1, 0))
        
        grad_Wp = grad_Wp / size_Wp
        grad_Wn = grad_Wn / size_Wn
        
        return grad_input, grad_weight, grad_Wp, grad_Wn, None, None, None # None car pas de gradient pour le threshold
        
class Conv2D_QuantModule(nn.Module):
    """ Personalized module for quantized convolution using the autograd function defined above """
    
    def __init__(self, ternary, in_channels, out_channels, kernel_size, stride, padding, threshold=0.05, Wp_init=1, Wn_init=-1):
        super().__init__()
        
        # weight initialisation, no bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        
        # Wp, WN = 1 at the beginning
        self.Wn = nn.Parameter(torch.tensor([Wn_init]).float())
        self.Wp = nn.Parameter(torch.tensor([Wp_init]).float())
        
        # Keeping threshold, stride and padding for the forward pass
        self.threshold = torch.tensor([threshold], requires_grad=False).cuda()
        self.stride = torch.tensor([stride]).float().cuda()
        self.padding = torch.tensor([padding]).float().cuda()

        self.function = Conv2D_QuantFunction_Ternary() if ternary else Conv2D_QuantFunction_Binary()
            
    def forward(self, X):
        output =  self.function.apply(X, self.weight, self.Wp, self.Wn, self.threshold, self.stride, self.padding)
        return output