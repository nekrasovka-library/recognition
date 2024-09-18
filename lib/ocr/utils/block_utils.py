import numpy as np

class Vertex:
    def __init__(self, value=None, parent=None, children=None):
        if children is None: children = []
        self.id = value[0]
        self.bbox = value[1:5]
        self.w_id = value[5]
        self.c_ids = value[6]
        self.children = children
        self.parent = parent
        
        self.a_c_ids = set(self.c_ids)
        
    def remove_a_c_ids(self, c_ids):
        for c_id in c_ids:
            self.a_c_ids.remove(c_id)
        
    def __repr__(self):
        return f'id: {self.id}; bbox: {self.bbox}, w_id: {self.w_id}, c_ids: {self.c_ids}, a_ids: {self.a_c_ids}'
    
    def __hash__(self):
        return hash(self.id)


def get_column_number(real_bboxes, W, columns=list(range(1,22)), 
                      left_margin_scale=0.6,
                      right_margin_scale=1.1,
                      min_width_scale=0.1,
                      max_width_scale=None,
                     ):
    widths = np.array([el[2] - el[0] for el in real_bboxes])
    
    probs = []; Ws = []
    for delimeter in columns:
        www = []
        for el in widths:
            if (left_margin_scale*W/delimeter < el < right_margin_scale*W/delimeter):
                www.append(0)
            elif (min_width_scale is not None) and (el < min_width_scale*W/delimeter):
                pass
            elif (max_width_scale is not None) and (el > max_width_scale*W/delimeter):
                pass
            else:
                www.append(1)
                
        hist = np.histogram(www, bins=2, density=False)
        prob = hist[0][0] / (hist[0].sum() + 1e-7)
        probs.append(prob)
        Ws.append(hist[1][0])

    culumn_number = columns[np.argmax(probs)]
    return columns, probs, culumn_number

def dfs_generator(start, use_sort=False):
    yield start
    upcoming = start.children.copy()
    if use_sort:
        upcoming = sorted(upcoming, key=lambda x: x.bbox[0])
    while len(upcoming) > 0:
        node = upcoming.pop(0)
        yield from dfs_generator(node)


def calculate_iosu(boxes1, boxes2, major='first'):
    '''
    boxes1 shape = (N, 4)
    boxes2 shape = (M, 4)
    
    iou shape = (M, N)
    iou.argmax(0) shape = N
    
    example: boxes1, boxes2 = np_bbox_line, np_bbox_block


    '''

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
    
    if major == 'first':
        iosu = (interArea / (boxAArea + 1e-7)).T
    elif major == 'second':
        iosu = (interArea / (boxBArea + 1e-7))
    else:
        raise NotImplementedError()

    return iosu


def agregate_bboxes(bboxes, column_w, column_w_min=None, column_w_max=None):
    '''
    Args:
        bboxes - list of bboxes or 2d np.array with the same axis order
    Return:
        list_of_vertexes - sorted list of class Vertex objects
    '''

    if column_w_min is None:
        column_w_min = 0.6 * column_w
    if column_w_max is None:
        column_w_max = 1.1 * column_w

    list_of_vertexes = []
    
    for idx, el in enumerate(bboxes):
        _data = [idx] + list(el)
        x0, y0, x1, y1 = el
        el_w = x1 - x0
        w_id   = int(el_w // column_w)
        col_id = int(x1 // column_w)

        rem_w = el_w % column_w 
        rem_c = x1 % column_w 

        if rem_w > (column_w_max - column_w):
            w_id += 1
        if rem_c > (column_w_max - column_w):
            col_id += 1

        if w_id >= 1:
            col_id -= w_id
            col_id = [col_id]
            for _ in range(w_id-1):
                col_id.append(col_id[-1]+1)
        else:
            col_id = [col_id]

        col_id = tuple(col_id)

        _data.append(w_id)
        _data.append(col_id)

        _data_t = tuple(_data)
        vertex = Vertex(_data_t)
        list_of_vertexes.append(vertex)
        
    list_of_vertexes = sorted(list_of_vertexes, key=lambda x: x.bbox[1])
    list_of_vertexes = sorted(list_of_vertexes, key=lambda x: x.c_ids[0])
    
    return list_of_vertexes


def could_be_matched(block_a, block_d):
    for block_a_c_id in block_a.c_ids:
        if block_a_c_id not in block_d.a_c_ids:
#             print(block_a.id, ' rej -> rej ', block_d.id, block_a_c_id, ' not in ', block_d.a_c_ids)
            return False
    return True

def build_linked_component(col, blocks_in_columns, block_idx, used):
    root = blocks_in_columns[col][block_idx]
    if root.id in used: return None
    used.add(root.id)
    for _col in root.c_ids:
        current_idx = -1
        for idx, el in enumerate(blocks_in_columns[_col]):
            if el.id == root.id:
                current_idx = idx + 1
                break
        assert current_idx != -1, 'assert current_idx != -1'
        
        if current_idx < len(blocks_in_columns[_col]):
            block = blocks_in_columns[_col][current_idx]
#             print(block.id, ' -> ', root.id)
            if block.id not in used:
                if could_be_matched(block, root):
#                     print(True)
                    root.children.append(block)
                    root.remove_a_c_ids(block.c_ids)
                    block.parent = root
                    
                    _ = build_linked_component(_col, blocks_in_columns, current_idx, used)
    return root


def filer_agregate_bboxes(list_of_vertexes):
    _bboxes = np.array([np.array(el.bbox) for el in list_of_vertexes])
    ious = calculate_iosu(_bboxes, _bboxes)
    ious -= np.diag([1 for _ in range(ious.shape[0])])
    _valid = ious.max(1) < 0.5

    filtered_list_of_vertexes = [el for idx, el in enumerate(list_of_vertexes) if _valid[idx]]
    return filtered_list_of_vertexes


def get_blocks_in_columns(COLUMN_N, filtered_list_of_vertexes):
    blocks_in_columns = []
    for col in range(COLUMN_N):
        for block in filtered_list_of_vertexes:
            if col in block.c_ids:
                if len(blocks_in_columns) < (col + 1):
                    blocks_in_columns.append([])
                blocks_in_columns[col].append(block)
        if len(blocks_in_columns) < (col + 1):
            blocks_in_columns.append([])
    for col in range(COLUMN_N):
        if len(blocks_in_columns[col]) > 0:
            blocks_in_columns[col] = sorted(blocks_in_columns[col], key=lambda x: x.bbox[1])
    return blocks_in_columns

def build_structure(blocks_in_columns):
    roots  = []
    active_leaves = []
    used = set()

    for col in range(len(blocks_in_columns)):
        for block_idx in range(len(blocks_in_columns[col])):
            root = build_linked_component(col, blocks_in_columns, block_idx, used)
            if root is not None:
                roots.append(root)
    return roots

def get_roots_v1(COLUMN_N, COLUMN_W, real_bboxes):
    list_of_vertexes = agregate_bboxes(real_bboxes, COLUMN_W)
    filtered_list_of_vertexes = filer_agregate_bboxes(list_of_vertexes)
    blocks_in_columns = get_blocks_in_columns(COLUMN_N, filtered_list_of_vertexes)
    roots = build_structure(blocks_in_columns)
    return roots


def node_to_dict(node, ocr, chain_id, reading_id):
    data = {
        'text'       : ocr[node.id],
        'id'         : node.id,
        'bbox'       : [int(el) for el in node.bbox],
        'chain_id'   : chain_id,
        'reading_id' : reading_id,
        'children'   : [el.id for el in node.children],  
    }
    return data

def roots_to_result(roots, ocr):
    result = []
    for root_idx, root in enumerate(roots):
        result.append([])
        for node_idx, node in enumerate(dfs_generator(root)):
            result[root_idx].append(
                node_to_dict(node, ocr, root_idx, node_idx)
            )
    result = sorted(result, key=lambda x: x[0]['bbox'][0])
    result = sorted(result, key=lambda x: x[0]['bbox'][1])
    return result


def blocks_result_2_raw_text(blocks_result, split_symbol='\t'):
    text = None
    for chain in blocks_result:
        for chain_el in chain:
            if text is None:
                text = ''
            else:
                text += split_symbol
            text += chain_el['text']
    if text is None:
        text = ''
    return text


def get_roots_v2(COLUMN_N, COLUMN_W, real_bboxes):
    list_of_vertexes = agregate_bboxes(real_bboxes, COLUMN_W)
    filtered_list_of_vertexes = filer_agregate_bboxes(list_of_vertexes)
    list_of_vertexes_by_width = sorted(filtered_list_of_vertexes,
                                       key=lambda x: (x.bbox[2] - x.bbox[0]))
    
    roots = [el for el in list_of_vertexes_by_width]

    for block in list_of_vertexes_by_width:
        _dists, _min_dist = calc_dist(block, list_of_vertexes_by_width)
        min_idx = np.argsort(_dists)[0]
        min_dist = _dists[min_idx]
        if min_dist is not np.nan:
            if _min_dist > 0 and min_dist > _min_dist:
                continue
            parent_block = list_of_vertexes_by_width[min_idx]

            parent_block.children.append(block)
            block.parent = parent_block

            roots.remove(block)
    return roots

def calc_dist(core_block, blocks, margine_scale=0.3):
    xa0, ya0, xa1, ya1 = core_block.bbox
    h, w = ya1 - ya0, xa1 - xa0
    
    distancies = []
    
    min_dist = -1
    
    for block in blocks:
        xb0, yb0, xb1, yb1 = block.bbox
        bh, bw = yb1 - yb0, xb1 - xb0
        
        if block.id == core_block.id:
            distancies.append(np.nan)
            continue
    
        x_min = xb0 - margine_scale * w
        x_max = xb1 + margine_scale * w
        
        x_min_b = xa0 - margine_scale * bw
        x_max_B = xa1 + margine_scale * bw
        dist = ya0 - yb1
        
        if (xb0 > x_min_b) and (xb1 < x_max_B) and (ya0 >= yb0):
            if min_dist < 0:
                min_dist = dist
            else:
                min_dist = min(min_dist, dist)
        
        if (xa0 < x_min) or (xa1 > x_max) or (ya0 < yb0):
            distancies.append(np.nan)
        else:
            distancies.append(dist)
    return distancies, min_dist

