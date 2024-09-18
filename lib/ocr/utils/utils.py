import os, sys
import cv2
import numpy as np

from skimage.transform import resize

from math import ceil

def data_to_text(data):
    res = ''
    last_symbol = '0'
    for el in data:
        if el is None:
            res += ' '
            last_symbol = '0'
        else:
            if last_symbol in '!),.:;?]}~_':
                res += ' '
            symbol = el[0]
            res += symbol
            last_symbol = symbol
    return res

def write_html_with_cnn_flag(img_path, data, W, H, output_name='index', additional_html_code=None):
        with open("{}.html".format(output_name), "w", encoding='utf-8') as html:
            html.write('''<html>
    <head>
        <meta charset="utf-8">
            <style>
                .l{position:absolute;color:rgba(0,100,0,0.9);border:1px solid rgba(255,0,0,0.3);overflow:hidden;}
                body{background-color: #FFDDBB;}
            </style>
    </head>
        <body>
<div style="position:relative;text-align:left;color:#000;">
<img src="img_path" alt="Snow" style="position:left;_width:img_Wpx;_height:img_Hpx;">\n'''.replace('img_path', img_path).replace('img_H', str(H)).replace('img_W', str(W)))

            coeff = 1.0;
            for letter in data:
                if letter is None:
                    continue
                colour = '255, 0, 0' if letter[6] else '0, 0, 255'
                # colour = '255, 0, 0' if letter[8] == 1 else '0, 0, 255'
                symbol = letter[0].upper() if letter[7] else letter[0].lower()
                # print(letter)

                html_symbol = """<div class="l" style="border-color: rgb({0}); color: rgb(255,255,255); left:{1};bottom: {2}; width: {3}; height: {4}; font-size: {5}" title="{6}">{7}</div>\n""".format(
                    colour, int(letter[1]*coeff), int(letter[2]*coeff), int(letter[3]*coeff), int(letter[5]*coeff), int(letter[5]*coeff), letter, symbol)
                html.write(html_symbol)
            html.write('\n')
            if additional_html_code is not None:
                html.write(additional_html_code)
            html.write("""</div></body></html>""")

def softmax(x, axis=-1):
    numerator = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
    out = numerator / np.expand_dims(np.sum(numerator, axis=axis), -1)
    return out


def scale(x, _min=None, _max=None, return_max_min=False):
    if _min is None and _max is None:
        _min = np.min(x); _max = np.max(x)
    if return_max_min:
        return (x - _min) / (_max - _min + 1e-7), _min, _max
    return (x - _min) / (_max - _min + 1e-7)

def BoxesIntersect(box1, box2):
    if box1['x'] + box1['w'] < box2['x']            : return False
    if box1['x']             > box2['x'] + box2['w']: return False
    if box1['y'] + box1['h'] < box2['y']            : return False
    if box1['y']             > box2['y'] + box2['h']: return False
    return True

# def scale(x):
#     _min = np.min(x); _max = np.max(x)
#     return (x - _min) / (_max - _min)

def rgb2grayscale(x_batch):
    if len(x_batch.shape) == 4:
        return 0.2989 * x_batch[:,:,:,0:1] + 0.5870 * x_batch[:,:,:,1:2] + 0.1140 * x_batch[:,:,:,2:3]
    else:
        return 0.2989 * x_batch[:,:,0] + 0.5870 * x_batch[:,:,1] + 0.1140 * x_batch[:,:,2]

def resize_img(image, shape=(320, 320), mode='constant'):
    w, h, layers = image.shape
    if shape == (w, h):
        return image
    local_image = []
    for l_idx in range(layers):
        image_padded = np.pad(image[:,:,l_idx], ((max(max(w, h) - w, 0), 0), (max(max(w, h) - h, 0), 0)), mode=mode)
        local_image.append(np.expand_dims(resize(image_padded, shape, mode=mode), -1))
    return np.concatenate(local_image, axis=-1)

def resize_batch(self, img_batch, shape, mode='constant'):
    new_batch = []
    for image_idx in range(img_batch.shape[0]):
        new_batch.append(np.expand_dims(resize_img(img_batch[image_idx], shape), 0 ))
    return np.concatenate(new_batch, axis=0)

def _resize(image, shape, mode='wrap'):
    from skimage.transform import resize
    result = []
    b, w, h, layers = image.shape
    for b_idx in range(b):
        local_image = []
        for l_idx in range(layers):
            local_image.append(np.expand_dims(resize(image[b_idx, :, :, l_idx], shape, mode=mode), -1))
        result.append(np.expand_dims(np.concatenate(local_image, axis=-1), 0))
    result = np.concatenate(result, axis=0)
    return result

# def softmax(x):
#     numerator = np.exp(x - np.max(x))
#     out = numerator / np.expand_dims(numerator.sum(-1), -1)
#     return out


def draw_symbol_bbox(mask, bbox):
    _left, _top, _right, _bottom = bbox

    x = _left
    y = _top
    w = _right - _left
    h = _bottom - _top

    _edges = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

    cv2.line(mask, _edges[0], _edges[1], (255,255,255), 2)
    cv2.line(mask, _edges[0], _edges[2], (255,255,255), 2)
    cv2.line(mask, _edges[3], _edges[1], (255,255,255), 2)
    cv2.line(mask, _edges[3], _edges[2], (255,255,255), 2)


def save_data(filename, data, return_data_flag=True):
    res = ''
    return_data = []
    for el in data:
        if el is not None:
            return_data.append(el)
            res += '_|_'.join(map(str, el)) + '\n'
    with open(filename, "w", encoding='utf-8') as text_file:
        text_file.write(res)
    if return_data_flag:
        return return_data


def load_data(filename):
    with open(filename, "r", encoding='utf-8') as text_file:
        data = text_file.readlines()
        
    improve_type = lambda s : (str(s[0]),int(s[1]),np.int64(s[2]),np.int64(s[3]),int(s[4]),int(s[5]),bool(s[6]),\
                                              bool(s[7]),int(s[8]),float(s[9]))
    data = [improve_type(el.replace('\n', '').split('_|_')) for el in data]
    return data


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bbox2np(bboxes):
    np_bboxes = []
    for el in bboxes:
        _bbox = el[1]
        np_bboxes.append([_bbox['x'], _bbox['y'], _bbox['x'] + _bbox['w'], _bbox['y'] + _bbox['h']])
    return np.array(np_bboxes)


def calculate_iou(boxes1, boxes2):
    '''
    boxes1 shape = (N, 4)
    boxes2 shape = (M, 4)
    
    iou shape = (M, N)
    iou.argmax(0) shape = N
    
    example: boxes1, boxes2 = np_bbox_line, np_bbox_block


    '''

    if (len(boxes1.shape) == 1) or (len(boxes2.shape) == 1):
        return None

    x11, y11, x12, y12 = boxes1[:,0], boxes1[:,1], boxes1[:,2], boxes1[:,3]
    
    x21 = np.repeat(boxes2[:,0][:, None], x11.shape[0], axis=1)
    y21 = np.repeat(boxes2[:,1][:, None], x11.shape[0], axis=1)
    x22 = np.repeat(boxes2[:,2][:, None], x11.shape[0], axis=1)
    y22 = np.repeat(boxes2[:,3][:, None], x11.shape[0], axis=1)
    
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-7)
    return iou

