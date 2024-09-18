from ocr.ocr4 import OCR4 as OCR
from ocr.ocr_corrector import Corrector as Corrector
from image_cutter.model import Model as ImageCutter

import os, sys, json, ntpath
from .utils.utils import str2bool, save_data, load_data, write_html_with_cnn_flag, data_to_text
from .utils.block_utils import blocks_result_2_raw_text
from basemodel import BaseModel

class Model(BaseModel):
    name             = 'TextRecognition'
    _dir             = os.path.dirname(os.path.realpath(__file__))
    _init_inputs     = set()
    _inputs          = {'page_path', 'output_dir',}
    _optional_inputs = {'use_shared_bottom', 'image_preprocessing_flag', 'copy_image', 'collect_images'
                        'allowed_extentions', 'N_JOBS', 'LAST_IDX', 'pointsize', 'page_image_url'}
    _outputs         = {'status',}
    
    _description = {
            'version': 'v5.0.0',
            'ocrType': 'Tsseract4LSTM',
            'saveImage': False,
            'correctionCNN': False,
            'correctionBayes': True,
            'imageCutter': True,
            'segmentationLine': True,
            'segmentationBlock': True,
            'BlockLinking' : True,
            }

    def __init__(self, N_JOBS=1,
                 logger=None, tqdm_type='default'):
        super().__init__(logger, tqdm_type)
        self.set_n_jobs(N_JOBS)
        self.ocr = OCR(logger=self.logger, tqdm_type=tqdm_type)

        self.corrector = Corrector(logger=self.logger, tqdm_type=tqdm_type)
        # self.corrector.create_N_gramm('1grams-3.txt')
        # self.corrector.dump_N_gramm()
        self.corrector.load_N_gramm()

        self.image_cutter = ImageCutter(logger=self.logger, tqdm_type=tqdm_type)

    def run(self, *, page_path, output_dir,
            use_shared_bottom=True, image_preprocessing_flag=False,
            copy_image=False, LAST_IDX=None, collect_images=False,
            allowed_extentions={'.jpg', '.tif'}, N_JOBS=1, pointsize=25, page_image_url=None):
        self.set_n_jobs(N_JOBS)

        self.logger.info('page_image_url: {}'.format(page_image_url))
        self.logger.info('page_path: {}'.format(page_path))
        self.logger.info('output_dir: {}'.format(output_dir))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename, extention = os.path.splitext(ntpath.basename(page_path))
        assert extention in allowed_extentions, 'extention shoud be in allowed_extentions'

        self.logger.info('Start: ocr recognition [image_regognize]'.format())
        preprocessed_image, symbol_result, line_result, block_result, ocrResult, W, H, blocks_result_v1, blocks_result_v2 = \
            self.ocr.image_regognize(page_path,
                                image_preprocessing_flag=image_preprocessing_flag,
                                last_idx=LAST_IDX,
                                use_shared_bottom=use_shared_bottom,
                                collect_images=collect_images
                                )

        if copy_image:
            os.system('cp {} {}'.format(page_path, os.path.join(output_dir, filename + extention)))

        block_data  = self.ocr.generate_data_to_plot(block_result, H, pointsize=pointsize)
        line_data   = self.ocr.generate_data_to_plot(line_result, H, pointsize=pointsize)
        symbol_data = self.ocr.generate_data_to_plot(symbol_result, H, pointsize=pointsize)

        self.logger.info('start image_cutter...'.format())
        image_cutter_result = self.image_cutter.img_process(
            os.path.join(output_dir, 'images'), page_path)

        additional_html_code = ''
        # image_cutter_template = '<img src="{}" style="position:absolute; TOP:{}px; LEFT:{}px; WIDTH:{}px; HEIGHT:{}px">\n'
        # for _image_path, image_coord in image_cutter_result:
        #     additional_html_code += image_cutter_template.format(
        #         os.path.join('images', ntpath.basename(_image_path)),
        #         image_coord['y'], image_coord['x'], image_coord['w'],image_coord['h'])
        # with open(os.path.join(output_dir, 'images/info.json'), 'w') as outfile:
        #     json.dump(image_cutter_result, outfile, separators=(',\n', ' : '))
        # self.logger.info('done image_cutter'.format())
        
        out_page = os.path.join(output_dir, 'data')

        save_data(out_page + '.txt', symbol_data, return_data_flag=False)

        self.logger.info('Start - correctorion...'.format())
        data_corrected = self.corrector.correction(symbol_data)
        self.logger.info('Done - correctorion...'.format())

        save_data(out_page + '_corrected.txt', data_corrected, return_data_flag=False)

        if page_image_url is None:
            self.ocr.write_html_with_cnn_flag(img_path=filename + extention, data=block_data, W=W, H=H, output_name=out_page + '_block', additional_html_code=additional_html_code)
            self.ocr.write_html_with_cnn_flag(img_path=filename + extention, data=line_data, W=W, H=H, output_name=out_page + '_line', additional_html_code=additional_html_code)
            self.ocr.write_html_with_cnn_flag(img_path=filename + extention, data=symbol_data, W=W, H=H, output_name=out_page + '_symbol', additional_html_code=additional_html_code)
            self.ocr.write_html_with_cnn_flag(img_path=filename + extention, data=data_corrected, W=W, H=H, output_name=out_page + '_symbol_corrected', additional_html_code=additional_html_code)
        else:
            self.ocr.write_html_with_cnn_flag(img_path=page_image_url, data=block_data, W=W, H=H, output_name=out_page + '_block', additional_html_code=additional_html_code)
            self.ocr.write_html_with_cnn_flag(img_path=page_image_url, data=line_data, W=W, H=H, output_name=out_page + '_line', additional_html_code=additional_html_code)
            self.ocr.write_html_with_cnn_flag(img_path=page_image_url, data=symbol_data, W=W, H=H, output_name=out_page + '_symbol', additional_html_code=additional_html_code)
            self.ocr.write_html_with_cnn_flag(img_path=page_image_url, data=data_corrected, W=W, H=H, output_name=out_page + '_symbol_corrected', additional_html_code=additional_html_code)

        with open(os.path.join(output_dir, 'description.json'), 'w') as f:
            json.dump(self._description, f)

        with open(os.path.join(output_dir, 'blocks_result_v1.json'), 'w') as f:
            json.dump(blocks_result_v1, f)

        with open(os.path.join(output_dir, 'blocks_result_v2.json'), 'w') as f:
            json.dump(blocks_result_v2, f)

        with open(os.path.join(output_dir, 'ocrResult.txt'), 'w') as f:
            f.writelines(ocrResult)

        ocrResult_blocks_v1 = blocks_result_2_raw_text(blocks_result_v1)
        ocrResult_blocks_v2 = blocks_result_2_raw_text(blocks_result_v2)

        with open(os.path.join(output_dir, 'ocrResult_blocks_v1.txt'), 'w') as f:
            f.writelines(ocrResult_blocks_v1)

        with open(os.path.join(output_dir, 'ocrResult_blocks_v2.txt'), 'w') as f:
            f.writelines(ocrResult_blocks_v2)

        return {'status' : True, 'text': ocrResult_blocks_v2,
                'data' : data_to_text(symbol_data), 'data_corrected' : data_to_text(data_corrected)}

