from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import collections
from glob import glob
import os
from PIL import Image

"""
main.py 함수를 참고하여 다음을 생각해봅시다.

1. CRNN_dataset은 어떤 모듈을 상속받아야 할까요?
    : Dataset (torch.utils.data.Dataset)
2. CRNN_dataset의 역할은 무엇일까요? 왜 필요할까요?
    : CRNN 모델 input dataset의 형태를 만져주는 역할을 한다.
    단순 Dataset 모듈을 이용하는게 아닌 오버라이드하여 label의 길이와 이미지 사이즈를 CRNN 모델에 맞춰주기 위해 필요하다.
3. 1.의 모듈을 상속받는다면 __init__, __len__, __getitem__을 필수로 구현해야 합니다. 각 함수의 역할을 설명해주세요.
    1) __init__ : dataset 형성에 필요한 변수 자리 확보 및 전처리
    2) __len__ : dataset의 길이, 즉 이미지의 개수
    3) __getitem__ : idx번째 샘플을 가져오는 인덱싱

"""


class CRNN_dataset(Dataset):
    def __init__(self, path, w=100, h=32, alphabet='0123456789abcdefghijklmnopqrstuvwxyz', max_len=36):
        self.max_len=max_len
        self.path = path
        self.files = glob(path+'/*.jpg') 
        self.n_image = len(self.files)
        assert (self.n_image > 0), "해당 경로에 파일이 없습니다. :)"

        self.transform = transforms.Compose([
            transforms.Resize((w,h)), # image 사이즈를 w, h를 활용하여 바꿔주세요.
            transforms.ToTensor() # tensor로 변환해주세요.
        ])
        """
        strLabelConverter의 역할을 설명해주세요.
        1. text 문제를 풀기 위해 해당 함수는 어떤 역할을 하고 있을까요?

        target text와 length의 Tensor 변환과 같은 전처리 역할

        2. encode, decode의 역할 설명

        encode는 label의 text와 length를 Tensor 타입으로 변환시켜주는 역할을 한다. 한 마디로 str -> Tensor 라 할 수 있다.
        decode는 이와 반대로 Tensor 타입이었던 결과물을 다시 str 타입으로 변환시켜주는 역할을 한다.

        """
        self.converter = strLabelConverter(alphabet) 
        
    def __len__(self):
        return self.n_image # hint: __init__에 정의한 변수 중 하나

    def __getitem__(self,idx):
        label = self.files[idx].split('_')[1]
        img = Image.open(self.files[idx]).convert('L')
        img = self.transform(img)
        """
        max_len이 왜 필요할까요? # hint: text data라는 점

        label text의 길이를 맞춰주기 위해서 필요하다.
        max_len 보다 크다면 crop하여 제한시키고, 작다면 빈자리를 채워 길이를 맞춰준다.

        """

        if len(label) > self.max_len:
            label = label[:self.max_len]
        label_text, label_length = self.converter.encode(label)

        if len(label_text) < self.max_len:
            temp = torch.ones(self.max_len-len(label), dtype=torch.int)
            label_text = torch.cat([label_text, temp])

        return img, (label_text, label_length) # hint: main.py를 보면 알 수 있어요 :)



# 아래 함수는 건드리지 마시고, 그냥 쓰세요 :)
class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-' 

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
