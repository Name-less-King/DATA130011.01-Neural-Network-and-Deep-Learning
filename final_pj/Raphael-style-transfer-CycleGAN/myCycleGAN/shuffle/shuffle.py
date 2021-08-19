import time
import torch
import matplotlib.pyplot as plt 

PATCH = 121

def shuffle_patch(s):
    input_size = s.shape[2:]
    kernel_size = (PATCH,PATCH)
    padding_module = torch.nn.ReflectionPad2d(PATCH//2)
    s = padding_module(s)
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=PATCH)
    patches = unfold(s)
    n_patches = patches.shape[2]
    perm = torch.randperm(n_patches)
    patches = patches[:,:,perm]
    
    fold = torch.nn.Fold(output_size=input_size, kernel_size=kernel_size, stride=PATCH,padding=PATCH//2)
    s = fold(patches)
    return s 

if __name__ == '__main__':
    style_path = 'cycleset/trainA/16. 335x476.jpg'
    style = plt.imread(style_path)
    style = torch.Tensor(style).unsqueeze(0)
    style = style.permute(0,3,1,2)
    shuffled = shuffle_patch(style)
    shuffled = shuffled.permute(0,2,3,1)
    shuffled = shuffled.numpy().squeeze(0)

    plt.imshow(shuffled/255)
    plt.axis('off')
    plt.savefig('shuffled.png')