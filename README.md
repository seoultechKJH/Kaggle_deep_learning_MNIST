# Kaggle_deep_learning_MNIST
Kaggle에서 제공되는 MNIST 이미지 분류 data에 대하여, Pytorch를 통한 딥러닝 Feedforward Neural Network 모델 설계 및 분석


# Dataset
- torchvision dataset의 Fashion MNIST dataset을 사용했으며, 60,000개의 training set과 10,000개의 test set을 통해 모델 구축 및 검증을 진행함
- 각 샘플은 28 × 28 pixel(=784 pixel)의 흑백이미지이며, 10개의 클래스가 라벨링되어 있음
- 각 pixel 값은 0~255 사이의 값을 가지고 10개의 클래스는 다음과 같음 (0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot)
- 이에 대하여 각 클래스에 따라 데이터를 분류하기 위해, pytorch를 활용하여 logistic regression 모델 1개와 FNN(Feedforward Neural Network) 모델 3개를 구축하고 성능평가를 수행함


# Model 1 (Logistic regression)
- 다중분류 logistic regression에서는 전체 확률의 합이 1이 되도록 산출하는 softmax unit을 사용해야 함
- (Cost function) 연산과정에 softmax 함수를 포함하고 있는 cross entropy를 사용함
- (Output Unit) class 개수와 동일하게 10으로 설정
- (Learning rate) 0.001로 설정하여 미세 조정하며 탐색하도록 함
- (epoch) 10번 수행


# Model 2 (Feedforward Neural Network)
- (Cost function) 세 가지 모델 모두 신경망의 출력값이 확률일 때 사용하는 cross entropy를 사용하여 최적의 parameter를 찾도록 설정함
- (Hidden layer) 세 가지 모델 모두 2개의 hidden layer를 설정
- (Output Unit) class 개수와 동일하게 10으로 설정
- (Hidden Unit) 세 가지 모델에 따라 차이를 둠
> (Model 1) 784 (Input) -> 16 (Hidden 1) -> 32 (Hidden 2) -> 10 (Output)
