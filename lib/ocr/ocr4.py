import numpy as np
import os, sys, json
import scipy.stats as st
import cv2

from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, iterate_level, OEM, PSM
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm
from time import time
import logging


from .utils.utils import BoxesIntersect, draw_symbol_bbox, softmax, scale, bbox2np, calculate_iou
from .utils.block_utils import get_column_number, roots_to_result, get_roots_v1, get_roots_v2
from basemodel import BaseModel

class OCR4(BaseModel):
    name = 'OCR'
    _dir = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, logger=None, tqdm_type='default'):
        super().__init__(logger, tqdm_type)
        
    def read(self, input_filename, image_preprocessing_flag=False):
        self.logger.info('Reading: {}'.format(input_filename))
        image = Image.open(input_filename)
        W, H  = image.size
        image_np = np.array(image)
        if image_preprocessing_flag:
            self.logger.info('image_preprocessing...'.format())
            image_np = self.image_preprocessing(image_np)
        image = Image.fromarray(image_np)
        return image, image_np, W, H
    
    def image_preprocessing(self, img):
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        # result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        return result_norm
    
    def image_regognize(self, page, image_preprocessing_flag=False, last_idx=np.inf,
                        use_shared_bottom=True, collect_images=False,
                        is_high_available=True, debug_output=False):
        if last_idx is None: last_idx=np.inf
        self.logger.info('Start - image reading...'.format())
        _start_time = time()
        image, image_np, W, H = self.read(page, image_preprocessing_flag=image_preprocessing_flag)
        self.logger.info('Done - image reading [s:{}]'.format(time()-_start_time))
        
        with PyTessBaseAPI(oem=OEM.LSTM_ONLY,  lang='rus+eng') as api:
            self.logger.info('SetImage to api'.format())
            api.SetImage(image)
            _start_time = time()
            block_boxes = api.GetComponentImages(RIL.BLOCK, True)
            self.logger.info('Found {} textblock image components [s:{}]'.format(len(block_boxes), time()-_start_time))
            line_boxes  = api.GetComponentImages(RIL.TEXTLINE, True)
            _start_time = time()
            self.logger.info('Found {} textline image components [s:{}]'.format(len(line_boxes), time()-_start_time))
            
            self.logger.info('Start recognition...'.format())
            _start_time = time()
            ocrResult = api.GetUTF8Text()
            self.logger.info('Done - recognized [s:{}]'.format(time()-_start_time))

            self.logger.info('Start bbox processing...'.format())
            _bbox_processing_start_time = time()
            
            real_bboxes = np.array([np.array([
                el[1]['x'], el[1]['y'], el[1]['x']+el[1]['w'], el[1]['y']+el[1]['h']]) for el in block_boxes])
            
            bbox_ocrResult = []
            _start_time = time()
            self.logger.info('Start bbox recognition...'.format())
            with PyTessBaseAPI(oem=OEM.LSTM_ONLY,  lang='rus+eng') as api2:
                api2.SetImage(image)
                _block_boxes = api2.GetComponentImages(RIL.BLOCK, True)
                for bbox in tqdm(real_bboxes):
                    x0, y0, x1, y1 = bbox
                    api2.SetRectangle(x0, y0, x1-x0, y1-y0)
                    text = api2.GetUTF8Text()
                    bbox_ocrResult.append(text)
            self.logger.info('Done - bbox recognition [s:{}], len bbox_ocrResult: {}'.format(time()-_start_time, len(bbox_ocrResult)))

            if len(real_bboxes) != 0:
                columns, probs, culumn_number = get_column_number(real_bboxes, W)
                self.logger.info('culumn_number: {}'.format(culumn_number))

                roots = get_roots_v1(culumn_number, W / culumn_number, real_bboxes.copy())
                blocks_result_v1 = roots_to_result(roots, bbox_ocrResult)

                roots = get_roots_v2(culumn_number, W / culumn_number, real_bboxes.copy())
                blocks_result_v2 = roots_to_result(roots, bbox_ocrResult)
            else:
                blocks_result_v1 = []
                blocks_result_v2 = []

            self.logger.info('Done - bbox processing [s:{}]'.format(time()-_bbox_processing_start_time))

            self.logger.info('Start - block processing...'.format())
            _start_time = time()
            level = RIL.BLOCK
            block_result = []
            if len(ocrResult) > 0:
                for block_idx, block_pointer in enumerate(iterate_level(api.GetIterator(), level=level)):
                    _text = block_pointer.GetUTF8Text(level)
                    _bbox = block_pointer.BoundingBox(level)
                    _conf = block_pointer.Confidence(level)
                    _im   = block_pointer.GetBinaryImage(level) if collect_images else None
                    block_result.append((_text, _bbox, _conf, _im, None, block_idx, None))
            self.logger.info('Done - block processing [s:{}]'.format(time()-_start_time))
            
            self.logger.info('Start - line processing...'.format())
            _start_time = time()
            level = RIL.TEXTLINE
            line_result = []
            if len(ocrResult) > 0:
                for line_idx, block_pointer in enumerate(iterate_level(api.GetIterator(), level=level)):
                    _text = block_pointer.GetUTF8Text(level)
                    _bbox = block_pointer.BoundingBox(level)
                    _conf = block_pointer.Confidence(level)
                    _im   = block_pointer.GetBinaryImage(level) if collect_images else None
                    line_result.append((_text, _bbox, _conf, _im, line_idx, None, None))
            self.logger.info('Done - line processing [s:{}]'.format(time()-_start_time))
            
            self.logger.info('Start - symbol processing...'.format())
            _start_time = time()
            level = RIL.SYMBOL
            level_elements = []
            level_bboxes   = []
            if len(ocrResult) > 0:
                for idx, symbol_pointer in enumerate(iterate_level(api.GetIterator(), level=level)):
                    if block_idx == last_idx: break
                    _text = symbol_pointer.GetUTF8Text(level)
                    _bbox = symbol_pointer.BoundingBox(level)
                    _conf = symbol_pointer.Confidence(level)
                    _im   = symbol_pointer.GetBinaryImage(level) if collect_images else None
                    
                    level_elements.append((_text, _bbox, _conf, _im))
                    level_bboxes.append(list(_bbox))
            self.logger.info('Done - symbol processing [s:{}]'.format(time()-_start_time))
            
            self.logger.info('Start - iou calculating...'.format())
            _start_time = time()
            np_block_boxes  = bbox2np(block_boxes)
            np_line_boxes   = bbox2np(line_boxes)
            np_level_bboxes = np.array(level_bboxes)  
            
            line_block_iou = calculate_iou(np_line_boxes, np_block_boxes)
            level_line_iou = calculate_iou(np_level_bboxes, np_line_boxes)
            self.logger.info('Done - iou calculating [s:{}]'.format(time()-_start_time))

            if line_block_iou is None or level_line_iou is None:
                symbol_result = []
                if debug_output:
                    return image, symbol_result, line_result, block_result, ocrResult, W, H, blocks_result_v1, blocks_result_v2, np_line_boxes, np_block_boxes
                return image, symbol_result, line_result, block_result, ocrResult, W, H, blocks_result_v1, blocks_result_v2

            line_block = line_block_iou.argmax(0)
            iindexes = np.array(list(enumerate(line_block))).T
            line_block[line_block_iou[iindexes[1], iindexes[0]] < 1e-2] = -1

            level_line = level_line_iou.argmax(0)
            iindexes = np.array(list(enumerate(level_line))).T
            level_line[level_line_iou[iindexes[1], iindexes[0]] < 1e-2] = -1
            
            level_line = sorted([[level_idx, line_idx] for level_idx, line_idx in enumerate(level_line)],
                                key=lambda x: x[1])
            
            self.logger.info('Start - line checking and blok, line, symbol matching...'.format())
            _start_time = time()
            symbol_result = []
            new_line = []
            last_line_idx = level_line[0][1]
            for level_idx, line_idx in level_line:
                _text, _bbox, _conf, _im = level_elements[level_idx]
                _line_idx = line_idx
                _block_idx = line_block[_line_idx]
                _capital = None
                
                if line_idx != last_line_idx:
                    if last_line_idx == -1:
                        self.line_checker(new_line, False, is_high_available)
                    else:
                        self.line_checker(new_line, use_shared_bottom, is_high_available)
                    symbol_result  += new_line
                    symbol_result.append(None)
                    new_line = []
                    last_line_idx = line_idx
                new_line.append([_text, _bbox, _conf, _im, _line_idx, _block_idx, _capital])
            if last_line_idx == -1:
                self.line_checker(new_line, False, is_high_available)
            else:
                self.line_checker(new_line, use_shared_bottom, is_high_available)
            symbol_result  += new_line
            symbol_result.append(None)
            self.logger.info('Done - line checking and block, line, symbol matching [s:{}]'.format(
                time()-_start_time))

            # self.logger.info('Start - blocks text matching with blocks...'.format())
            # _start_time = time()
            # if len(block_result) != len(block_boxes):
            #     self.logger.warning('len( block_result) != len(block_boxes) [{} != {}]'.format(
            #         len(block_result), len(block_boxes)))
            # else:
            #     for b1, b2 in zip(block_result, block_boxes):
            #         x1, y1 = b2['x'], b2['y']
            #         x2, y2 = x1 + b2['w'], y1 + b2['h']
            #         b2 = (x1, y1, x2, y2)
            #         if b1 != b2:
            #             self.logger.warning('blocks bboxes not matching ({} != {})'.format(b1, b2))
            #             break
            # self.logger.info('Done - blocks text matching with blocks [s:{}]'.format(time()-_start_time))
            
            if debug_output:
                return image, symbol_result, line_result, block_result, ocrResult, W, H, blocks_result_v1, blocks_result_v2, np_line_boxes, np_block_boxes
            return image, symbol_result, line_result, block_result, ocrResult, W, H, blocks_result_v1, blocks_result_v2
        
    def line_checker(self, line_result, use_shared_bottom=True, is_high_available=True):
        _tops = []; _bottoms = []
        _bottom_coord = None
        for el in line_result:
            if (el is not None) and (el is not False):
                _left, _top, _right, _bottom = el[1]
                _tops.append(-_top)
                _bottoms.append(_bottom)
        if len(_tops) > 0:
            _tops = scale(np.array(_tops).astype(np.float32))
            _mode = st.mode(_tops).mode
            _shift = 2.5 * np.std(_tops[_tops > _mode - 0.1])
        if len(_bottoms) > 0:
            _bottoms = np.array(_bottoms)
            _bottom_coord = st.mode(_bottoms).mode

        allow_capitel_letters_after = {171, 46,}
        # to check symbol in "allow_capitel_letters_after" use "ord(/u...)" and "chr(171)"
        idx = 0
        for el in line_result:
            if (el is not None) and (el is not False):
                _left, _top, _right, _bottom = el[1]
                if use_shared_bottom and (_bottom_coord is not None):
                    el[1] = (_left, _top, _right, _bottom_coord)
                else:
                    el[1] = (_left, _top, _right, _bottom)
                if _tops[idx] > (_mode + _shift) and is_high_available:
                    el[6] = True
                    is_high_available = True
                else:
                    el[6] = False
                    is_high_available = False
                if ord(el[0]) in allow_capitel_letters_after:
                    is_high_available = True
                idx += 1
            else:
                is_high_available = True
    
    def generate_data_to_plot(self, result, H, pointsize=24):
        '''el of result = [_text, _bbox, _conf, _im, _line_idx, _block_idx, _capital]'''
        data = []
        for el in result:
            if (el is not None) and (el is not False):
                text_part = el[0]
                block_idx = el[5]
                _left, _top, _right, _bottom = el[1]
                x = _left
                y = H - _bottom
                h = _bottom - _top
                w = _right - _left
                cnn_flag = False
                prob = el[2] / 100.
                _capital = el[6]
                data.append((text_part, x, y, h, w, pointsize, cnn_flag, _capital, block_idx, prob))
            else:
                data.append(None)
        return data

    def write_html_with_cnn_flag(self, img_path, data, W, H, output_name='index', additional_html_code=None):
        '''el of data = (text_part, x, y, h, w, pointsize, cnn_flag, _capital, block_idx, prob)'''
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
<img src="img_path" alt="Snow" style="position:left;_width:img_Wpx;_height:img_Hpx;">\n'''.replace(
                'img_path', img_path).replace('img_H', str(H)).replace('img_W', str(W)))

            coeff = 1.0;
            for text in data:
                if (text is None) or (text is False):
                    continue
                colour = '255, 0, 0' if text[6] else '0, 0, 255'
                # colour = '255, 0, 0' if letter[8] == 1 else '0, 0, 255'
                capital = text[7]
                if capital is None:
                    symbol = text[0]
                else:
                    symbol = text[0].upper() if capital else text[0].lower()
                # print(letter)

                html_symbol = """<div class="l" style="border-color: rgb({0}); color: rgb(255,255,255); left:{1};bottom: {2}; width: {3}; height: {4}; font-size: {5}" title="{6}">{7}</div>\n""".format(
                    colour, int(text[1]*coeff), int(text[2]*coeff), int(text[4]*coeff), int(text[3]*coeff), int(text[5]*coeff), text, symbol)
                html.write(html_symbol)
            html.write('\n')
            if additional_html_code is not None:
                html.write(additional_html_code)
            html.write("""</div></body></html>""")

