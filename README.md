# Kaggle_deep_learning_MNIST
Kaggle에서 제공되는 Fashion MNIST 이미지 분류 data에 대하여, Pytorch를 통한 딥러닝 Feedforward Neural Network 모델 설계 및 분석 - 테크니컬 포트폴리오 기술평가항목 : 1


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
  - (Model 1) 784 (Input) -> 16 (Hidden 1) -> 32 (Hidden 2) -> 10 (Output)
  - (Model 2) 784 (Input) -> 100 (Hidden 1) -> 100 (Hidden 2) -> 10 (Output)
  - (Model 3) 784 (Input) -> 526 (Hidden 1) -> 268 (Hidden 2) -> 10 (Output)
- (Learning rate) 모델별 차이를 두어, Model 1부터 3으로 갈수록 learning rate를 낮추어 gradient를 미세 조정하며 탐색하도록 함
  - (Model 1) 0.1 / (Model 2) 0.05 / (Model 3) 0.01
- (epoch) 세 가지 모델에 대해 동일하게 10번 수행함
- (Regularization) Model 1은 별도의 조작을 가하지 않고 수행했으며, Model 2에는 Data Augmentation, Model 3에서는 Model 2에 추가로 Dropout을 적용하여 변화를 보고자 함
  - (Data Augmentation) training data를 증가시켜 학습 성능을 좋게하고 overfitting을 방지할 수 있음. 해당 실험에서는 이미지 좌우를 뒤집는 가로 대칭이동방법(RandomHorizontalFlip)을 사용하여 training set을 2배 증가시킴
  - (Dropout) 학습 진행과정에서 신경망 모델의 일부를 사용하지 않으므로써 overfitting을 방지함


# Result
- logistic regression
  - 모델의 최종 loss가 0.895, accuracy가 0.715이며, Shirt에 대한 분류가 가장 어려움
- Feedforward Neural Network Model 1
  - 모델의 최종 loss가 0.453, accuracy가 0.838이며, Shirt에 대한 분류가 가장 어려움
- Feedforward Neural Network Model 2
  - 모델의 최종 loss가 0.408, accuracy가 0.854이며, Pullover에 대한 분류가 가장 어려움
- Feedforward Neural Network Model 3
  - 모델의 최종 loss가 0.477, accuracy가 0.833이며, Shirt에 대한 분류가 가장 어려움
- (결론) FNN Model 2가 accuracy 측면에서 가장 좋은 성능을 보였으며, Shirt class에 대한 분류가 가장 어려운 것으로 확인됨
