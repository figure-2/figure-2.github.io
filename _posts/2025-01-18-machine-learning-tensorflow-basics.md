---
title: TensorFlow 기본 모델링
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- TensorFlow
- 딥러닝
- Sequential API
- Functional API
- Subclassing API
- 콜백
- 커스텀손실함수
- 하이퍼파라미터튜닝
toc: true
date: 2023-09-16 11:00:00 +0900
comments: false
mermaid: true
math: true
---
# TensorFlow 기본 모델링

> 231016 학습한 내용 정리

## TensorFlow 개요

### 정의
- **TensorFlow**: Google에서 개발한 오픈소스 머신러닝 프레임워크
- 딥러닝 모델 구축과 훈련을 위한 종합적인 플랫폼
- **Tensor**: 다차원 배열, **Flow**: 데이터 흐름 그래프

### 특징
- **유연성**: 다양한 머신러닝 모델 구축 가능
- **확장성**: CPU, GPU, TPU에서 실행 가능
- **생태계**: 풍부한 라이브러리와 도구 제공
- **프로덕션**: 실제 서비스 배포에 최적화

### 장점
- **성능**: 높은 계산 성능
- **유연성**: 다양한 모델 아키텍처 지원
- **커뮤니티**: 활발한 커뮤니티와 문서
- **통합**: Keras와의 완벽한 통합

## TensorFlow 설치 및 기본 사용법

### 1. 설치
```bash
# CPU 버전
pip install tensorflow

# GPU 버전
pip install tensorflow-gpu

# 최신 버전
pip install tensorflow==2.12.0
```

### 2. 기본 사용법
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow 버전 확인
print(f"TensorFlow 버전: {tf.__version__}")

# 샘플 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TensorFlow 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test))

# 배치 크기 설정
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
```

## TensorFlow 모델 구축

### 1. Sequential API
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

# Sequential 모델 생성
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=[Accuracy()]
)

# 모델 구조 확인
model.summary()

# 모델 학습
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    verbose=1
)
```

### 2. Functional API
```python
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# Functional API 모델 생성
inputs = Input(shape=(20,))
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model_functional = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model_functional.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=[Accuracy()]
)

# 모델 학습
history_functional = model_functional.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    verbose=1
)
```

### 3. Subclassing API
```python
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

# 커스텀 레이어 정의
class CustomDense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(CustomDense, self).build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

# Subclassing 모델 정의
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = CustomDense(64, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = CustomDense(32, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = CustomDense(16, activation='relu')
        self.dense4 = CustomDense(1, activation='sigmoid')
    
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        return self.dense4(x)

# 모델 생성
model_subclassing = CustomModel()

# 모델 컴파일
model_subclassing.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=[Accuracy()]
)

# 모델 학습
history_subclassing = model_subclassing.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    verbose=1
)
```

## TensorFlow 고급 기능

### 1. 콜백 (Callbacks)
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 콜백 정의
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

# 콜백을 사용한 모델 학습
model_with_callbacks = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_with_callbacks.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=[Accuracy()]
)

history_callbacks = model_with_callbacks.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)
```

### 2. 커스텀 손실 함수
```python
# 커스텀 손실 함수 정의
def custom_loss(y_true, y_pred):
    """커스텀 손실 함수"""
    # 기본 이진 교차 엔트로피
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # L2 정규화
    l2_reg = tf.reduce_sum([tf.nn.l2_loss(w) for w in model.trainable_weights])
    
    return bce + 0.01 * l2_reg

# 커스텀 손실 함수를 사용한 모델
model_custom_loss = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_custom_loss.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=custom_loss,
    metrics=[Accuracy()]
)

history_custom_loss = model_custom_loss.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    verbose=1
)
```

### 3. 커스텀 메트릭
```python
# 커스텀 메트릭 정의
class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
        self.true_negatives.assign_add(tn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
        self.true_negatives.assign(0)

# 커스텀 메트릭을 사용한 모델
model_custom_metric = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_custom_metric.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=[CustomAccuracy()]
)

history_custom_metric = model_custom_metric.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    verbose=1
)
```

## TensorFlow 모델 평가 및 예측

### 1. 모델 평가
```python
# 모델 평가
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")

# 예측
predictions = model.predict(X_test_scaled)
predicted_classes = (predictions > 0.5).astype(int)

# 혼동 행렬
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, predicted_classes)
print("혼동 행렬:")
print(cm)

print("\n분류 보고서:")
print(classification_report(y_test, predicted_classes))
```

### 2. 학습 곡선 시각화
```python
# 학습 곡선 시각화
plt.figure(figsize=(15, 5))

# 손실 곡선
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('모델 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()
plt.grid(True)

# 정확도 곡선
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('모델 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend()
plt.grid(True)

# 예측 vs 실제
plt.subplot(1, 3, 3)
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', lw=2)
plt.title('예측 vs 실제')
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. 모델 저장 및 로드
```python
# 모델 저장
model.save('my_model.h5')

# 모델 로드
loaded_model = tf.keras.models.load_model('my_model.h5')

# 가중치만 저장
model.save_weights('my_weights.h5')

# 가중치만 로드
model.load_weights('my_weights.h5')

# 모델 구조 저장
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# 모델 구조 로드
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
```

## TensorFlow 실무 적용 예시

### 1. 회귀 문제
```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# 보스턴 주택 가격 데이터 로드
boston = load_boston()
X, y = boston.data, boston.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 회귀 모델 생성
regression_model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# 모델 컴파일
regression_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 모델 학습
history_regression = regression_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# 예측
predictions_regression = regression_model.predict(X_test_scaled)

# 성능 평가
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions_regression)
r2 = r2_score(y_test, predictions_regression)

print(f"평균제곱오차: {mse:.4f}")
print(f"결정계수: {r2:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_regression, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('회귀 예측 결과')
plt.grid(True)
plt.show()
```

### 2. 다중 클래스 분류
```python
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

# 아이리스 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 원-핫 인코딩
y_train_categorical = to_categorical(y_train, 3)
y_test_categorical = to_categorical(y_test, 3)

# 다중 클래스 분류 모델
multiclass_model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

# 모델 컴파일
multiclass_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
history_multiclass = multiclass_model.fit(
    X_train_scaled, y_train_categorical,
    epochs=100,
    validation_data=(X_test_scaled, y_test_categorical),
    verbose=1
)

# 예측
predictions_multiclass = multiclass_model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions_multiclass, axis=1)

# 성능 평가
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predicted_classes)
print(f"정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, predicted_classes, target_names=iris.target_names))
```

### 3. 하이퍼파라미터 튜닝
```python
import kerastuner as kt

# 하이퍼파라미터 튜닝 모델 정의
def build_model(hp):
    model = Sequential()
    
    # 입력 레이어
    model.add(Dense(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        activation='relu',
        input_shape=(20,)
    ))
    
    # 드롭아웃 레이어
    model.add(Dropout(
        rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    ))
    
    # 히든 레이어
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_{i+2}', min_value=16, max_value=64, step=16),
            activation='relu'
        ))
        model.add(Dropout(
            rate=hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.5, step=0.1)
        ))
    
    # 출력 레이어
    model.add(Dense(1, activation='sigmoid'))
    
    # 컴파일
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 하이퍼파라미터 튜너 생성
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='tuner_results',
    project_name='tensorflow_tuning'
)

# 하이퍼파라미터 탐색
tuner.search(
    X_train_scaled, y_train,
    epochs=50,
    validation_data=(X_test_scaled, y_test),
    callbacks=[EarlyStopping(patience=5)]
)

# 최적 모델 가져오기
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("최적 하이퍼파라미터:")
print(best_hyperparameters.values)
```

## 주의사항 및 모범 사례

### 1. 모델 설계
- **레이어 수**: 과적합을 방지하기 위해 적절한 레이어 수 선택
- **뉴런 수**: 데이터 복잡도에 맞는 뉴런 수 설정
- **활성화 함수**: 문제 유형에 맞는 활성화 함수 선택

### 2. 학습 최적화
- **학습률**: 적절한 학습률 설정
- **배치 크기**: 메모리와 성능을 고려한 배치 크기 선택
- **에포크 수**: 조기 종료를 통한 과적합 방지

### 3. 성능 모니터링
- **검증 데이터**: 검증 데이터를 통한 성능 모니터링
- **콜백**: 적절한 콜백을 통한 학습 제어
- **메트릭**: 문제에 맞는 메트릭 선택

## 마무리

TensorFlow는 딥러닝 모델 구축과 훈련을 위한 강력한 프레임워크입니다. Sequential API, Functional API, Subclassing API 등 다양한 모델 구축 방법을 제공하며, 콜백, 커스텀 손실 함수, 커스텀 메트릭 등 고급 기능을 통해 실무에서 요구되는 복잡한 모델을 구축할 수 있습니다. 적절한 모델 설계와 학습 최적화를 통해 높은 성능의 딥러닝 모델을 개발할 수 있습니다.
