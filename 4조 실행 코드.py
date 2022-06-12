## 서울전역 분류

# 패키지 불러오기
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# X 데이터(접수일시,거리) 추출
import csv
f = open('./data/소방서.csv', 'r', encoding='utf-8-sig')
reader = csv.reader(f)
x_line1 = []
x_line2 = []
for line in reader:
    x_line1.append(line[5])
    x_line2.append(line[2])
x_list_d = [x_line1, x_line2]
x_list = [list(i) for i in zip(*x_list_d)]

f = open('./data/x_data.csv', 'w', encoding='utf-8-sig', newline='')
writer = csv.writer(f)
for row in x_list:
    writer.writerow(row)
f.close()

# X 데이터 분류(전처리)
f = open('./data/x_data.csv', 'r', encoding='utf-8-sig')
reader = csv.reader(f)
f.readline()
lines = []
for i in reader:
    if 4 <= float(i[1]):
        i[1] = 4
    lines.append(i)
f = open('./data/x_Decisiontree.csv', 'w', encoding='utf-8-sig', newline='')
writer = csv.writer(f)
header = ['접수일시(00시00분을 0으로 하여 하루의 시작 기준 몇 분 지났는지)', '소방서거리(㎞)']
writer.writerow([i for i in header])
writer.writerows(lines)
f.close()

# X 데이터셋
x_data = pd.read_csv('./data/x_Decisiontree.csv')
x_data.astype(float)
X = np.array(x_data)

# y 데이터(걸린시간) 추출
f = open('./data/소방서.csv', 'r', encoding='utf-8-sig')
reader = csv.reader(f)
y_line = []
for line in reader:
    y_line.append(line[6])
y_list_d = [y_line]
y_list = [list(i) for i in zip(*y_list_d)]

f = open('./data/y_data.csv', 'w', encoding='utf-8-sig', newline='')
writer = csv.writer(f)
for row in y_list:
    writer.writerow(row)
f.close()

# y 데이터 분류(전처리)
f = open('./data/y_data.csv', 'r', encoding='utf-8-sig')
reader = csv.reader(f)
f.readline()
lines = []
for i in reader:
    if float(i[0]) <= 3:
        i[0] = 0
    elif 3 < float(i[0]) <= 4:
        i[0] = 1
    elif 4 < float(i[0]) <= 5:
        i[0] = 2
    elif 5 < float(i[0]):
        i[0] = 3
    lines.append(i)
f = open('./data/y_Decisiontree.csv', 'w', encoding='utf-8-sig', newline='')
writer = csv.writer(f)
header = ['접수로부터 도착까지 시간(숫자형식)']
writer.writerow([i for i in header])
writer.writerows(lines)
f.close()

# y 데이터셋
y_data = pd.read_csv('./data/y_Decisiontree.csv')
y_data.astype(float)
y = np.array(y_data)

# 훈련과 테스트모델 분리
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 데이터 학습,정확도 예측
dt_limit = DecisionTreeClassifier(max_depth = 3, random_state = 0)
dt_clf = dt_limit.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
dt_prediction = dt_clf.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, dt_prediction))

# 나뭇가지 그림
plt.figure(figsize=(30, 30))
plot_tree(dt_clf, filled=True)
plt.show()

# 결정경계
x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
Z = dt_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 분석 그림
plt.contourf(xx, yy, Z, cmap=plt.cm.jet)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0], s=10, edgecolor='black', cmap=plt.cm.jet)
plt.xlabel("Reception time")
plt.ylabel("Distance")
plt.title("Decision tree\n(Required time)\nThe whole area of Seoul")
plt.text(1900, 2, 'Below 3min : 0\n4min : 1\n5min : 2 \nAbove 6min : 3')
plt.colorbar()
plt.show()







## 구별 분류
df = pd.read_csv('./data/구별데이터.csv', header=None)
Guname_df = pd.read_csv('./data/구이름.csv', header=None)
Guname = np.array(Guname_df)
for i in range(25):
    # X 데이터셋
    globals()['xdf_{}'.format(i)] = df[[4*i+1,4*i+2]].dropna()
    globals()['xdf_{}'.format(i)].astype(float)
    globals()['X_{}'.format(i)] = np.array(globals()['xdf_{}'.format(i)])

    # y 데이터셋
    globals()['ydf_{}'.format(i)] = df[[4*i+3]].dropna()
    globals()['ydf_{}'.format(i)].astype(float)
    globals()['Y_{}'.format(i)] = np.array(globals()['ydf_{}'.format(i)])

    # y 데이터셋 전처리
    ynrow = globals()['Y_{}'.format(i)].shape[0]
    for j in range(ynrow):
        if globals()['Y_{}'.format(i)][j]<=3:
            globals()['Y_{}'.format(i)][j]=0
        elif 3 < globals()['Y_{}'.format(i)][j]<=4:
            globals()['Y_{}'.format(i)][j]=1
        elif 4 < globals()['Y_{}'.format(i)][j]<=5:
            globals()['Y_{}'.format(i)][j]=2
        elif 5 < globals()['Y_{}'.format(i)][j]:
            globals()['Y_{}'.format(i)][j]=3

    # 훈련과 테스트모델 분리
    globals()['X_train_{}'.format(i)], globals()['X_test_{}'.format(i)], globals()['Y_train_{}'.format(i)], \
    globals()['Y_test_{}'.format(i)] = train_test_split(globals()['X_{}'.format(i)], globals()['Y_{}'.format(i)])

    # 데이터 학습,정확도 예측
    dt_limit = DecisionTreeClassifier(max_depth = 3, random_state = 0)
    dt_clf_gu = dt_limit.fit(globals()['X_train_{}'.format(i)], globals()['Y_train_{}'.format(i)])
    from sklearn.metrics import accuracy_score
    dt_prediction_gu = dt_clf_gu.predict(globals()['X_test_{}'.format(i)])
    print('Accuracy: %.2f' % accuracy_score(globals()['Y_test_{}'.format(i)], dt_prediction_gu))

    # 나뭇가지 그림
    plt.figure(figsize=(30, 30))
    plot_tree(dt_clf_gu, filled=True)
    plt.show()

    # 결정경계
    x_min_gu, x_max_gu = globals()['X_train_{}'.format(i)][:, 0].min(), globals()['X_train_{}'.format(i)][:, 0].max()
    y_min_gu, y_max_gu = globals()['X_train_{}'.format(i)][:, 1].min(), globals()['X_train_{}'.format(i)][:, 1].max()
    xx_gu, yy_gu = np.meshgrid(np.arange(x_min_gu, x_max_gu, 0.02),
                         np.arange(y_min_gu, y_max_gu, 0.02))
    Z_gu = dt_clf_gu.predict(np.c_[xx_gu.ravel(), yy_gu.ravel()])
    Z_gu = Z_gu.reshape(xx_gu.shape)

    # 분석 그림
    plt.contourf(xx_gu, yy_gu, Z_gu, cmap=plt.cm.jet)
    plt.scatter(globals()['X_train_{}'.format(i)][:, 0], globals()['X_train_{}'.format(i)][:, 1],
                c=globals()['Y_train_{}'.format(i)][:, 0], s=10, edgecolor='black', cmap=plt.cm.jet)
    plt.title(Guname[i])
    plt.text(x_max_gu, 1600, 'Below 3min : 0\n3min - 4min : 1\n4min - 5min : 2 \nAbove 5min : 3')
    plt.xlabel("Distance")
    plt.ylabel("Reception time")
    plt.colorbar()
    plt.show()