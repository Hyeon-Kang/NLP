from sklearn.datasets import fetch_openml # Scikit-learn에서 제공하는 dataset이 모여있는 라이브러리
from sklearn.neural_network import MLPClassifier # DNN 모델 라이브러리
from sklearn.model_selection import train_test_split # train, test set을 구분하는 라이브러리

print("MNIST Data Downloading")
X, y = fetch_openml('mnist_784', version =1, return_X_y=True) # 2개의 변수에 image, label을 따로 저장 (X : image, y : label)
print("Success")
# fetch_openml 라이브러리에서 MNIST 28 X 28 의 픽셀을 가지는 데이터를 불러옴
# test_size =0.2 >> train : test = 8 : 2 (하나의 데이터셋에서 학습용, 검증용 구분)
# random_state

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

nn = MLPClassifier(hidden_layer_sizes=(128,128,128,128), max_iter=20, solver='sgd', learning_rate_init=0.001, verbose=True)
# 128,128,128,128 hidden layer shape, optimizer ; stochastic gradient descent

# MLPClassifier : MLP 레이어 선언
# hidden_layer_sizes : 128 > 128 > 128 > 128 > 10 레이어별 퍼셉트론 숫자
# solver = gradient descent optimizer 알고리즘 기입 (layer의 weigh[가중치]를 적정치로 조절해주는 알고리즘)
# learning_rate_init = 학습률

nn.fit(X_train, y_train)

print('Network Performance : %f'%nn.score(X_test, y_test))
