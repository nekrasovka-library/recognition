import os, sys, json
import numpy as np
from tqdm import tqdm
from time import sleep
import logging
from basemodel import BaseModel

class Corrector(BaseModel):
    name             = 'Corrector'
    _dir             = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, language_info_path='language_info',
                 logger=None, tqdm_type='default'):
        super().__init__(logger, tqdm_type)
        self.dir = os.path.join(self._dir, language_info_path)
        self.three_gramm_frequency = None
        self.all_three_gramm = None
        self.N = None
        
        
    def correct(self, word_letters, word_probs, N=3):
#         print(''.join(word_letters[N-1:]))
#         print(word_probs[N-1:])
        if len(word_letters) < (N - 1 + N):
            word_probs = [0 for el in range(len(word_letters))]
            return word_letters, word_probs
        for idx in range(0, len(word_letters) - N + 1):
            letter_prob = word_probs[idx+2]
            if not (1072 <= ord(word_letters[idx+2]) <= 1103):
                continue
            key1, key2, key3 = word_letters[idx:idx+3]
            probs = [(word_letters[idx + 2], 1e-5)]
            frequency_fixed = 1e-10
            if key1 in self.three_gramm_frequency and key2 in self.three_gramm_frequency[key1]:
                for new_letter in self.three_gramm_frequency[key1][key2]:
                    frequency_fixed += self.three_gramm_frequency[key1][key2][new_letter]
                prob_fixed = frequency_fixed / self.all_three_gramm
            else:
                prob_fixed = 1e-10

            if key1 in self.three_gramm_frequency and key2 in self.three_gramm_frequency[key1]:
                for new_letter in self.three_gramm_frequency[key1][key2]:

                    _fr = 0.
                    for _key1 in self.three_gramm_frequency:
                        for _key2 in self.three_gramm_frequency[_key1]:
                            if new_letter in self.three_gramm_frequency[_key1][_key2]:
                                _fr += self.three_gramm_frequency[_key1][_key2][new_letter]

                    p_b_a = self.three_gramm_frequency[key1][key2][new_letter] / _fr

                    if key3 == new_letter:
                        prob = letter_prob * p_b_a / prob_fixed
                    else:
                        prob = (1. - letter_prob) / 33. * p_b_a / prob_fixed
                    probs.append((new_letter, prob))
#             print(max(probs, key=lambda x: x[1])[0])
#             print(word_letters[idx + 2])
#             print(probs)
            _res = max(probs, key=lambda x: x[1])
            word_letters[idx + 2] = _res[0]
            word_probs[idx + 2] = _res[1]
#             print(''.join(word_letters[N-1:]))
#             print('='*10)
        return word_letters, word_probs


    def correction(self, data, N=3, use_tqdm=True):
        disable_tqdm = not use_tqdm
        data_corrected = []
        if len(data) == 0:
            return data_corrected
        word_letters = [None for _ in range(N-1)]
        word_probs   = [None for _ in range(N-1)]
        if (data[-1] is not None) or (data[-1] != False):
            data.append(None)
        
        _mark = False
        for idx in self.tqdm(range(len(data)), disable=disable_tqdm, miniters=len(data)*0.1):
        # for idx in range(len(data)):
            el = data[idx]
            if (el is not None) and (el is not False):
                _letter = el[0]; _prob   = el[9]
                word_letters.append(_letter)
                word_probs.append(_prob)
                _mark = True
            else:
                if _mark:
#                     print(''.join(word_letters[2:]))
                    word_letters, word_probs = self.correct(word_letters, word_probs, N)
#                     print(''.join(word_letters[2:]))
                    
                    for jdx in range(len(word_letters) - N + 1):
                        symbol, x, y, h, w, pointsize, cnn_flag, is_capital, block_id, prob = data[jdx + idx - len(word_letters) + N - 1]
                        symbol_new = word_letters[jdx + N - 1]
                        data_corrected.append((symbol_new, x, y, h, w, pointsize, cnn_flag, is_capital, block_id, prob))
#                 else:
                data_corrected.append(None)
                word_letters = [None for _ in range(N-1)]
                word_probs   = [None for _ in range(N-1)]
        return data_corrected
        
        
    def dump_N_gramm(self, name='N_gramm'):
        if self.three_gramm_frequency is not None:
            with open('{}/{}.json'.format(self.dir, name), 'w', encoding='utf-8') as outfile:
                json.dump(self.three_gramm_frequency, outfile, separators=(',\n', ' : '))
            with open('{}/{}.json'.format(self.dir, name + '_N'), 'w', encoding='utf-8') as outfile:
                json.dump({'all_three_gramm': self.all_three_gramm}, outfile, separators=(',\n', ' : '))

                
    def load_N_gramm(self, name='N_gramm'):
        with open('{}/{}.json'.format(self.dir, name), encoding='utf-8') as f:
            self.three_gramm_frequency = json.load(f)
        with open('{}/{}.json'.format(self.dir, name + '_N'), encoding='utf-8') as f:
            all_three_gramm = json.load(f)
        self.all_three_gramm = all_three_gramm['all_three_gramm']
            
        current_dict = self.three_gramm_frequency
        self.N = 0
        while type(current_dict) == dict:
            for key in current_dict:
                current_dict = current_dict[key]
                break
            self.N += 1
    
    def create_N_gramm(self, path, N=3, use_tqdm=True):
        disable_tqdm = not use_tqdm
        dir_1grams_txt = self.dir + '/' + path
        three_gramm_frequency = {}
        all_three_gramm = 0
        
        with open(dir_1grams_txt, encoding='utf-8') as f:
            for data in self.tqdm(f, total=1054210, disable=disable_tqdm):
                if len(data.split()) < 2:
                    continue
                _frequency, word = ' '.join(data.split()).split(' ') 
                _frequency = float(_frequency)
        #         _frequency = 1.

                word_utf8 = word
                word_utf8_list = [None for _ in range(N-1)] + list(word_utf8)
                _len_word_utf8 = len(word_utf8_list)
                if _len_word_utf8 >= N: #always True
                    for idx in range(0,_len_word_utf8 - N + 1):
                        new_three_gramm = three_gramm_frequency
                        for depth, letter in enumerate(word_utf8_list[idx:idx+N]):
                            if letter is not None:
                                letter = letter.lower()
                                if (ord(letter) < ord('а')) or (ord(letter) > ord('я')):
                                    if (ord(letter) < ord('0')) or (ord(letter) > ord('9')):
                                        letter = None
#                                 print(letter)
                            if depth != (N-1):
                                if letter not in new_three_gramm:
                                    new_three_gramm[letter] = {}
                                new_three_gramm = new_three_gramm[letter]
                            else:
                                if letter not in new_three_gramm:
                                    new_three_gramm[letter] = (_frequency)
                                else:
                                    new_three_gramm[letter] += _frequency
                                all_three_gramm += _frequency

        self.three_gramm_frequency = three_gramm_frequency
        self.all_three_gramm = all_three_gramm 
        self.N = N