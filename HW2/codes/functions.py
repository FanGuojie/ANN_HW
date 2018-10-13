import numpy as np
from scipy import signal
from utils import FZ

def conv2d_forward(input, W, b, kernel_size, pad,strides=1):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    (n,xc_in,xh_in,xw_in)=input.shape
    # print("input:",input.shape)
    (c_out,xc_in,xk,xk)=W.shape
    # print("W:",W.shape)
    conv_height=int((xh_in-xk+pad*2)/strides+1)
    conv_width=int((xw_in-xk+pad*2)/strides+1)
    output=np.zeros((n,c_out,conv_height,conv_width))
    for i in range(n):
        for j in range(c_out):
            temp=np.zeros((conv_height,conv_width))
            for k in range(xc_in):
                temp+=signal.convolve2d(input[i][k],W[j][k],'same')
            output[i][j]=temp
    # print("output",output.shape)
    return output


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    # print("input:",input.shape)
    # print("grad_output:",grad_output.shape)
    # print("b:",b.shape)
    # print("W:",W.shape)
    (n,c_in,h_in,w_in)=input.shape
    (n,c_out,h_out,w_out)=grad_output.shape
    (c_out,c_in,k,k)=W.shape
    grad_input=np.zeros((n,c_in,h_in,w_in))
    (c_out,)=b.shape
    grad_W=np.zeros((c_out,c_in,kernel_size,kernel_size))
    grad_b=np.zeros(c_out)
    # print("grad_W:",grad_W.shape)
    # print("grad_b:",grad_b.shape)
    # print("pad:",pad)
    avg_input=np.average(input,axis=1)
    avg_gradoutput=np.average(grad_output,axis=0)
    # print("avg_input:",avg_input.shape)
    for j in range(c_in):
        np.pad(avg_input[j], pad, 'constant', constant_values=0)
    for i in range(c_out):
        for j in range(c_in):
            grad_W[i][j]=signal.convolve2d(avg_input[j],avg_gradoutput[i],'valid')
    grad_b=np.sum(np.sum(avg_gradoutput,axis=2),axis=1)
    # print("grad_b value:",grad_b.shape)
    # print("grad_input:",grad_input.shape)
    for i in range(n):
        for j in range(c_in):
            temp=np.zeros((h_in,w_in))
            for k in range(c_out):
                temp+=signal.convolve2d(grad_output[i][k],FZ(W[k][j]),'same')
            grad_input[i][j]=np.matmul(temp,input[i][j])
    return grad_input,grad_W,grad_b



def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    (n,xc_in,xh_in,xw_in)=input.shape
    avg_height=int((xh_in-kernel_size+2*pad)/kernel_size)+1
    avg_width=int((xw_in-kernel_size+2*pad)/kernel_size)+1
    output=np.zeros((n,xc_in,avg_height,avg_width))
    # print("input:",input.shape)
    # print("avg_height",avg_height)
    # print("avg_width",avg_width)
    # print("k_size",kernel_size)


    for i in range(n):
        for j in range(xc_in):
            for k in range(avg_height):
                for l in range(avg_width):
                    output[i][j][k][l]=np.average(np.mat(input[i][j])[k*kernel_size:(k+1)*kernel_size,l*kernel_size:(l+1)*kernel_size])
                    # print(output[i][j][k][l])
    return output
def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    (n,xc_in,xh_in,xw_in)=input.shape
    (n,xc_in,xh_out,xw_out)=grad_output.shape

    grad_input=np.zeros((n,xc_in,xh_in,xw_in))
    k_square=kernel_size*kernel_size
    # print("grad_output:",grad_output)
    for i in range(n):
        for j in range(xc_in):
            for k in range(xh_out):
                for l in range(xw_out):
                    grad_input[i,j,k*kernel_size:(k+1)*kernel_size,l*kernel_size:(l+1)*kernel_size]=grad_output[i,j,k,l]/k_square
    # print("grad_input:",grad_input)
    return grad_input


