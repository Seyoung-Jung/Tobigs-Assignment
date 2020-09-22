from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from models import CRNN
from utils import CRNN_dataset, strLabelConverter
from tqdm import tqdm
import argparse
import os


def hyperparameters() :
    """
    argparse는 하이퍼파라미터 설정, 모델 배포 등을 위해 매우 편리한 기능을 제공합니다.
    파이썬 파일을 실행함과 동시에 사용자의 입력에 따라 변수값을 설정할 수 있게 도와줍니다.

    argparse를 공부하여, 아래에 나온 argument를 받을 수 있게 채워주세요.
    해당 변수들은 모델 구현에 사용됩니다.

    ---변수명---
    변수명에 맞춰 type, help, default value 등을 커스텀해주세요 :)
    
    또한, argparse는 숨겨진 기능이 지이이이인짜 많은데, 다양하게 사용해주시면 우수과제로 가게 됩니다 ㅎㅎ
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default = '.\dataset', help='Location of dataset')
    parser.add_argument('--savepath', type=str, default='best model', help='File name for saving best model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning Rate')
    parser.add_argument('--device', type=int, default=1, help='GPU Number')
    parser.add_argument('--img_width', type=int, default=100, help='Width of input image')
    parser.add_argument('--img_height', type=int, default=32, help='Height of input image')

    return parser.parse_args()


def main():
    args = hyperparameters()


    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')

    # gpu or cpu 설정
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu') 

    # train dataset load
    train_dataset = CRNN_dataset(path=train_path, w=args.img_width, h=args.img_height)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # test dataset load
    test_dataset = CRNN_dataset(path=test_path, w=args.img_width, h=args.img_height)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    

    # model 정의
    model = CRNN(args.img_height, 1, 37, 256)  # nc=1, nclass=37, nh=256
 
    # loss 정의
    criterion = nn.CTCLoss()
    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.999))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        assert False, "옵티마이저를 다시 입력해주세요. :("

    model = model.to(device)
    best_test_loss = 100000000

    for i in range(args.epochs):
        
        print('epochs: ', i)

        print("<----training---->")
        model.train()
        for inputs, targets in tqdm(train_dataloader):
            inputs = inputs.permute(0,1,3,2) # inputs의 dimension을 (batch, channel, h, w)로 바꿔주세요. hint: pytorch tensor에 제공되는 함수 사용
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs)
            preds = F.log_softmax(preds, dim=-1)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))


            """
            CTCLoss의 설명과 해당 로스의 input에 대해 설명해주세요.

            CTC(Connectionist Temporal Classification)이란, 입력 프레임 시퀀스와 타겟 시퀀스 간에
            명시적으로 할당해주지 않아도 모델을 학습할 수 있는 기법을 말한다. 
            CRNN을 살펴보면, 입력 이미지 feature vector sequence의 길이는 가변적이고 실제 단어의 글자수와도 맞지 않는다.
            기존의 CNN은 라벨 할당으로 학습한 것과 달리, 입력 sequence가 주어졌을 때 각 시점별로 본래 label sequence로 향하는 모든 가능한 경로를 고려하여 우도를 구하여 학습한다. 
            연산량의 감소를 위해 dynamic programming (앞에서 계산한 경로의 우도를 기억해두는 방법) 알고리즘을 활용한다는 특징이 있고,
            CTC layer는 RNN 출력 확률 벡터 sequence를 입력받아 loss를 계산하여 grandient를 통해 학습을 가능하게 만든다.
            
            loss의 input은 RNN layer의 출력 확률 벡터 sequence라고 할 수 있다.

            """

            loss = criterion(preds, target_text, preds_length, target_length) / batch_size
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        print("\n<----evaluation---->")

        """
        model.train(), model.eval()의 차이에 대해 설명해주세요.
        .eval()을 하는 이유가 무엇일까요?
        train은 말 그대로 학습 모드 , eval은 test 모드를 의미한다. 학습이 끝났으니 test 모드에 들어가자~! 하고 모델에게 알려주는 것이다.

        """

        model.eval() 
        loss = 0.0

        for inputs, targets in tqdm(test_dataloader):
            inputs = inputs.permute(0,1,3,2)
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs)
            preds = F.log_softmax(preds, dim=-1)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            loss += criterion(preds, target_text, preds_length, target_length) / batch_size # 학습이 아니라 test loss이니 밑에서 찍으려면 이 한 줄이 더 있어야 한다.


        print("\ntest loss: ", loss)
        if loss < best_test_loss:
            # loss가 bset_test_loss보다 작다면 지금의 loss가 best loss가 되겠죠?
            best_test_loss = loss
            # args.savepath을 이용하여 best model 저장하기
            torch.save(model.state_dict(), args.savepath)
            print("best model 저장 성공")



if __name__=="__main__":
    main()
