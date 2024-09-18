from tqdm import tqdm as tqdm
# from tqdm import tqdm_notebook as tqdm
import cv2, os
import logging, json
from basemodel import BaseModel

class Model(BaseModel):
    name             = 'ImageCutter'
    _dir             = os.path.dirname(os.path.realpath(__file__))
    _init_inputs     = set()
    _inputs          = {'page_path', 'output_dir',}
    _optional_inputs = {'thresh', 'maxval', 'N_JOBS',}
    _outputs         = set()

    def __init__(self, min_area_pct = 0.0000001,
                 max_hw_pct=0.9, min_hw_pct=0.1,
                 canny_threshold = 0, erosion_size = 20,
                 dilation_size = 5, thresh=150, maxval=255, N_JOBS=1,
                 logger=None, tqdm_type='default'):
        super().__init__(logger, tqdm_type)
        self.set_n_jobs(N_JOBS)

        self.canny_threshold = canny_threshold
        self.erosion_size    = erosion_size
        self.dilation_size   = dilation_size
        
        self.thresh = thresh
        self.maxval = maxval
        
        self.dilation_element_name = 'rectangle'
        self.erosion_element_name  = 'rectangle'
        
        self.min_area_pct = min_area_pct
        
        self.max_hw_pct = max_hw_pct
        self.min_hw_pct = min_hw_pct
        
        self.MORPH_TYPES = {
            "cross": cv2.MORPH_CROSS,
            "ellipse": cv2.MORPH_ELLIPSE,
            "rectangle": cv2.MORPH_RECT,
        }

    def run(self, page_path, output_dir,
            thresh=150, maxval=255,
            N_JOBS=1):
        self.set_n_jobs(N_JOBS)
        
        self.logger.info('page_path: {}'.format(page_path))
        self.logger.info('output_dir: {}'.format(output_dir))
        self.logger.info('thresh: {}'.format(thresh))
        self.logger.info('maxval: {}'.format(maxval))
        self.logger.info('self.n_jobs: {}'.format(self.n_jobs))

        result = self.img_process(output_dir, page_path,
                                  thresh=thresh, maxval=maxval,
                                  return_images=False)
        with open(os.path.join(output_dir, 'info.json'), 'w') as outfile:
            json.dump(result, outfile, separators=(',\n', ' : '))
        return {'status' : True, 'result' : result}

    
    def img_process(self, output_dir, page_path, thresh=150, maxval=255, return_images=False):
        # self.logger.info('img_process: {}'.format(page_path))
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        page      = cv2.imread(page_path)
        gray_page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        
        threshold_rc, threshold_image = cv2.threshold(gray_page, thresh, maxval, cv2.THRESH_BINARY)
        output_base_image = cv2.bitwise_not(threshold_image)
        output_image = output_base_image.copy()
        
        if self.dilation_size > 0:
#             print("Dilation %s at %d" % (element_name, dilation_size))
            element_name = self.dilation_element_name; element = self.MORPH_TYPES[element_name]
            structuring_element = cv2.getStructuringElement(element, (2 * self.dilation_size + 1,
                                                                      2 * self.dilation_size + 1))
            output_image = cv2.dilate(output_image, structuring_element)

        if self.erosion_size > 0:
#             print(("Erosion %s at %d" % (element_name, erosion_size)))
            element_name = self.erosion_element_name; element = self.MORPH_TYPES[element_name]
            structuring_element = cv2.getStructuringElement(element, (2 * self.erosion_size + 1,
                                                                      2 * self.erosion_size + 1))
            output_image = cv2.erode(output_image, structuring_element)

        if self.canny_threshold > 0:
#             print("Canny at %d" % canny_threshold)
            output_image = cv2.Canny(output_image, self.canny_threshold, self.canny_threshold * 3, 12)

        contours, hierarchy = cv2.findContours(output_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area_pct = self.min_area_pct
        min_area = min_area_pct * page.size

        max_height = int(round(self.max_hw_pct * page.shape[0]))
        max_width  = int(round(self.max_hw_pct * page.shape[1]))
        min_height = int(round(self.min_hw_pct * page.shape[0]))
        min_width  = int(round(self.min_hw_pct * page.shape[1]))

        EXTRACTED = []
        for idx, contour in enumerate(contours):
            length = cv2.arcLength(contours[idx], False)
            area   = cv2.contourArea(contours[idx], False)

            if area < min_area:
                continue

            poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
            x, y, w, h = cv2.boundingRect(poly)
            bbox = ((x, y), (x + w, y + h))

            if w > max_width or w < min_width or h > max_height or h < min_height:
#                 print("\t\t{0}: failed min/max check: {1}".format(idx, str(bbox)))
                continue

#             print("\t\t%4d: %16.2f%16.2f bounding box=%s" % (idx, length, area, bbox))

            extracted = (page[y:y + h, x:x + w], os.path.join(output_dir, '{}.jpg'.format(idx)), {'x':x, 'y':y, 'w':w, 'h':h})
            EXTRACTED.append(extracted)
            if output_dir is not None:
                cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(idx)), extracted[0])
        if return_images:
            return EXTRACTED
        return [[el[1], el[2]] for el in EXTRACTED]
        
    def dir_process(self, input_dir, output_dir, allowed_extentions={'.jpg', '.tif'}, return_images=False, use_tqdm=True):
        disable_tqdm = not use_tqdm
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        pages = os.listdir(input_dir)
        X = [(os.path.join(input_dir, page), os.path.splitext(page)[0])
             for page in pages if os.path.splitext(page)[1] in allowed_extentions]
        X = sorted(X)
#         return X
        if self.n_jobs == 1:
            _statuses = [self.img_process(os.path.join(output_dir, page_name),
                                          page_path, self.thresh, self.maxval,
                                          return_images) for page_path, page_name in tqdm(X, disable=disable_tqdm)]
        else:
            from joblib import Parallel, delayed
            _statuses = Parallel(n_jobs=self.n_jobs, verbose=True)(delayed(self.img_process)\
                (os.path.join(output_dir, page_name), page_path, self.thresh, self.maxval, return_images)\
                for page_path, page_name in tqdm(X, disable=disable_tqdm))
        statuses = []
        for _status in _statuses:
            statuses += _status
        if (len(statuses) > 0) and (len(statuses[0]) == 3):
            statuses_to_save = [[el[1], el[2]] for el in statuses]
        else:
            statuses_to_save = statuses
        with open('{}/{}.json'.format(output_dir, 'statuses'), 'w') as outfile:
            json.dump(statuses_to_save, outfile, separators=(',\n', ' : '))
        return _statuses
