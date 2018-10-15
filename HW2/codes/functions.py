import numpy as np


def conv2d_forward(input, W, b, kernel_size, pad, strides=1):
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
    (n, h_in, w_in, c_in) = input.shape
    (k, k, c_in, c_out) = W.shape
    conv_height = (h_in - k + pad * 2) // strides + 1
    conv_width = (w_in - k + pad * 2) // strides + 1
    output = np.zeros((n, conv_height, conv_width, c_out))
    input_with_pad = np.zeros([n, h_in + pad * 2, w_in + pad * 2, c_in])
    input_with_pad[:, pad: pad + conv_height, pad: pad + conv_width, :] = input
    output += b.reshape((1, 1, 1, c_out))
    for kh in range(k):
        for kw in range(k):
            output += np.dot(input_with_pad[:, kh: kh + conv_height, kw: kw + conv_width, :].reshape(-1, c_in),
                             W[kh, kw, :, :].reshape(c_in, c_out)).reshape(output.shape)
    return output, input_with_pad


def conv2d_backward(input_with_pad, grad_output, W, b, kernel_size, pad):
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
    (n, h_in_with_pad, w_in_with_pad, c_in) = input_with_pad.shape
    h_in = h_in_with_pad - 2 * pad
    w_in = w_in_with_pad - 2 * pad
    (n, h_out, w_out, c_out) = grad_output.shape
    (k, k, c_in, c_out) = W.shape
    grad_W = np.zeros((k, k, c_in, c_out))
    grad_b = np.sum(grad_output, axis=(0, 1, 2))
    for kh in range(k):
        for kw in range(k):
            grad_W[kh, kw, :, :] = np.dot(input_with_pad[:, kh:kh + h_out,
                                          kw:kw + w_out, :].reshape(-1, c_in).T,
                                          grad_output.reshape(-1, c_out))
    grad_input_with_pad = np.zeros(input_with_pad.shape)
    for kh in range(k):
        for kw in range(k):
            grad_input_with_pad[:, kh:kh + h_out, kw:kw + w_out, :] += \
                np.dot(grad_output.reshape(-1, c_out),W[kh, kw, :, :].reshape(c_out, c_in)).reshape(n, h_out, w_out, c_in)
    grad_input = grad_input_with_pad[:, pad:pad + h_in, pad:pad + w_in, :]

    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad=0):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    (n, h_in, w_in, c_in) = input.shape
    avg_height = (h_in + 2 * pad) // kernel_size
    avg_width = (w_in + 2 * pad) // kernel_size
    output = np.zeros((n, avg_height, avg_width, c_in))

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            output += input[:, kh::kernel_size, kw::kernel_size, :]
    output /= kernel_size * kernel_size
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad=0):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    (n, h_in, w_in, c_in) = input.shape
    grad_input = np.zeros(input.shape)
    grad_output *=  kernel_size * kernel_size
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            grad_input[:, kh::kernel_size, kw::kernel_size, :] = grad_output

    return grad_input
