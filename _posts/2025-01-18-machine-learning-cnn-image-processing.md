---
title: CNN (Convolutional Neural Network) 이미지 처리
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- CNN
- 합성곱신경망
- 이미지처리
- 데이터증강
- 전이학습
- 특징맵시각화
- Grad-CAM
- 성능최적화
toc: true
date: 2023-09-16 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# CNN (Convolutional Neural Network) 이미지 처리

> 231017 학습한 내용 정리

## CNN 개요

### 정의
- **Convolutional Neural Network** (합성곱 신경망)
- 이미지 처리에 특화된 딥러닝 모델
- 합성곱 연산을 통해 이미지의 지역적 특징을 추출

### 특징
- **지역적 특징**: 이미지의 지역적 패턴을 효과적으로 학습
- **가중치 공유**: 동일한 필터를 전체 이미지에 적용
- **계층적 학습**: 저수준 특징부터 고수준 특징까지 계층적 학습
- **변환 불변성**: 위치에 상관없이 동일한 특징 인식

### 장점
- **이미지 특화**: 이미지 데이터에 최적화된 구조
- **효율성**: 가중치 공유로 파라미터 수 감소
- **성능**: 이미지 분류에서 뛰어난 성능
- **확장성**: 다양한 이미지 크기에 적용 가능

## CNN 기본 구조

### 1. 합성곱 레이어 (Convolutional Layer)
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 데이터 로드
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 데이터 전처리
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 기본 CNN 모델
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
model.summary()
```

### 2. 풀링 레이어 (Pooling Layer)
```python
# 다양한 풀링 레이어 비교
def create_pooling_models():
    models = {}
    
    # Max Pooling
    models['max_pool'] = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    # Average Pooling
    models['avg_pool'] = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    # Global Average Pooling
    models['global_avg_pool'] = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    
    return models

# 풀링 모델들 생성
pooling_models = create_pooling_models()

# 각 모델 컴파일
for name, model in pooling_models.items():
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"{name} 모델 파라미터 수: {model.count_params()}")
```

### 3. 드롭아웃과 배치 정규화
```python
# 드롭아웃과 배치 정규화를 포함한 CNN 모델
model_advanced = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model_advanced.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
model_advanced.summary()
```

## CNN 고급 기법

### 1. 데이터 증강 (Data Augmentation)
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# 데이터 증강을 적용한 모델 학습
model_augmented = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_augmented.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 데이터 증강을 사용한 학습
history_augmented = model_augmented.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)
```

### 2. 전이 학습 (Transfer Learning)
```python
from tensorflow.keras.applications import VGG16, ResNet50

# VGG16 전이 학습
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # 기본 모델의 가중치 고정
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# ResNet50 전이 학습
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # 기본 모델의 가중치 고정
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# 전이 학습 모델들 생성
vgg16_model = create_vgg16_model()
resnet50_model = create_resnet50_model()

# 모델 컴파일
vgg16_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

resnet50_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("VGG16 모델 파라미터 수:", vgg16_model.count_params())
print("ResNet50 모델 파라미터 수:", resnet50_model.count_params())
```

### 3. 커스텀 CNN 아키텍처
```python
# 커스텀 CNN 아키텍처
class CustomCNN(models.Model):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # 특징 추출 부분
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.25)
        
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.Dropout(0.25)
        
        self.conv5 = layers.Conv2D(128, (3, 3), activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.25)
        
        # 분류 부분
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv5(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)
        x = self.dense2(x)
        
        return x

# 커스텀 모델 생성
custom_model = CustomCNN(num_classes=10)
custom_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
custom_model.build((None, 32, 32, 3))
custom_model.summary()
```

## CNN 실무 적용 예시

### 1. 이미지 분류
```python
# 이미지 분류 모델 학습
def train_image_classifier():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# 모델 학습
trained_model, training_history = train_image_classifier()

# 성능 평가
test_loss, test_accuracy = trained_model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도: {test_accuracy:.4f}")

# 예측
predictions = trained_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 혼동 행렬
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(true_classes, predicted_classes)
print("혼동 행렬:")
print(cm)

# 분류 보고서
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
print("\n분류 보고서:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))
```

### 2. 특징 맵 시각화
```python
# 특징 맵 시각화
def visualize_feature_maps(model, image, layer_names):
    """특징 맵을 시각화하는 함수"""
    # 중간 레이어의 출력을 가져오는 모델 생성
    intermediate_outputs = [model.get_layer(name).output for name in layer_names]
    intermediate_model = models.Model(inputs=model.input, outputs=intermediate_outputs)
    
    # 특징 맵 추출
    feature_maps = intermediate_model.predict(np.expand_dims(image, axis=0))
    
    # 시각화
    fig, axes = plt.subplots(len(layer_names), 1, figsize=(15, 5 * len(layer_names)))
    if len(layer_names) == 1:
        axes = [axes]
    
    for i, (layer_name, feature_map) in enumerate(zip(layer_names, feature_maps)):
        # 첫 번째 채널의 특징 맵만 시각화
        axes[i].imshow(feature_map[0, :, :, 0], cmap='viridis')
        axes[i].set_title(f'{layer_name} - 첫 번째 채널')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 샘플 이미지 선택
sample_image = X_test[0]

# 특징 맵 시각화
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2']
visualize_feature_maps(trained_model, sample_image, layer_names)
```

### 3. Grad-CAM 시각화
```python
# Grad-CAM 구현
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Grad-CAM 히트맵 생성"""
    # 마지막 합성곱 레이어와 분류 레이어를 포함한 모델 생성
    grad_model = models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # GradientTape를 사용하여 그래디언트 계산
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # 마지막 합성곱 레이어에 대한 그래디언트 계산
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # 각 채널의 중요도를 나타내는 벡터 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 마지막 합성곱 레이어의 출력에 중요도 가중치 적용
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 히트맵 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Grad-CAM 시각화
def display_gradcam(img, heatmap, alpha=0.4):
    """Grad-CAM 결과 시각화"""
    # 히트맵을 원본 이미지 크기로 리사이즈
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], [img.shape[0], img.shape[1]])
    heatmap = tf.squeeze(heatmap)
    
    # 히트맵을 컬러맵으로 변환
    heatmap = plt.cm.jet(heatmap)[..., :3]
    
    # 원본 이미지와 히트맵 결합
    superimposed_img = heatmap * alpha + img
    superimposed_img = tf.clip_by_value(superimposed_img, 0, 1)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM 히트맵')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title('결합된 이미지')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Grad-CAM 적용
sample_image = X_test[0]
heatmap = make_gradcam_heatmap(
    np.expand_dims(sample_image, axis=0), 
    trained_model, 
    'conv2d_2'
)
display_gradcam(sample_image, heatmap)
```

## CNN 성능 최적화

### 1. 하이퍼파라미터 튜닝
```python
import kerastuner as kt

def build_cnn_model(hp):
    """하이퍼파라미터 튜닝을 위한 CNN 모델"""
    model = models.Sequential()
    
    # 입력 레이어
    model.add(layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(32, 32, 3)
    ))
    
    # 풀링 레이어
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 두 번째 합성곱 레이어
    model.add(layers.Conv2D(
        filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=64),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        activation='relu'
    ))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 드롭아웃
    model.add(layers.Dropout(
        rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    ))
    
    # 완전 연결 레이어
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
        activation='relu'
    ))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    # 컴파일
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 하이퍼파라미터 튜너 생성
tuner = kt.Hyperband(
    build_cnn_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='cnn_tuning',
    project_name='cnn_hyperparameter_tuning'
)

# 하이퍼파라미터 탐색
tuner.search(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# 최적 모델 가져오기
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("최적 하이퍼파라미터:")
print(best_hyperparameters.values)
```

### 2. 모델 앙상블
```python
# 여러 CNN 모델 앙상블
def create_ensemble_models():
    """앙상블을 위한 여러 CNN 모델 생성"""
    models_list = []
    
    # 모델 1: 기본 CNN
    model1 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    models_list.append(model1)
    
    # 모델 2: 더 깊은 CNN
    model2 = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    models_list.append(model2)
    
    # 모델 3: 배치 정규화 포함
    model3 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    models_list.append(model3)
    
    return models_list

# 앙상블 모델들 생성
ensemble_models = create_ensemble_models()

# 각 모델 컴파일 및 학습
for i, model in enumerate(ensemble_models):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"모델 {i+1} 학습 중...")
    model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        verbose=0
    )

# 앙상블 예측
def ensemble_predict(models, X):
    """앙상블 모델들의 예측 결과 평균"""
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # 예측 결과 평균
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# 앙상블 예측
ensemble_predictions = ensemble_predict(ensemble_models, X_test)
ensemble_predicted_classes = np.argmax(ensemble_predictions, axis=1)

# 앙상블 성능 평가
ensemble_accuracy = np.mean(ensemble_predicted_classes == np.argmax(y_test, axis=1))
print(f"앙상블 정확도: {ensemble_accuracy:.4f}")

# 개별 모델 성능과 비교
individual_accuracies = []
for model in ensemble_models:
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    individual_accuracies.append(accuracy)

print("개별 모델 정확도:")
for i, acc in enumerate(individual_accuracies):
    print(f"모델 {i+1}: {acc:.4f}")
print(f"앙상블 정확도: {ensemble_accuracy:.4f}")
```

## 주의사항 및 모범 사례

### 1. 모델 설계
- **레이어 수**: 과적합을 방지하기 위해 적절한 레이어 수 선택
- **필터 크기**: 3x3 필터가 일반적으로 효과적
- **풀링**: Max Pooling이 일반적으로 Average Pooling보다 성능이 좋음

### 2. 학습 최적화
- **데이터 증강**: 과적합 방지와 성능 향상에 효과적
- **배치 정규화**: 학습 안정성과 성능 향상
- **드롭아웃**: 과적합 방지에 효과적

### 3. 성능 모니터링
- **검증 데이터**: 검증 데이터를 통한 성능 모니터링
- **콜백**: 조기 종료와 학습률 감소 활용
- **시각화**: 특징 맵과 Grad-CAM을 통한 모델 해석

## 마무리

CNN은 이미지 처리에 특화된 딥러닝 모델로, 합성곱 연산을 통해 이미지의 지역적 특징을 효과적으로 추출할 수 있습니다. 적절한 모델 설계, 데이터 증강, 전이 학습 등의 기법을 활용하여 높은 성능의 이미지 분류 모델을 구축할 수 있습니다. 또한 특징 맵 시각화와 Grad-CAM을 통해 모델의 동작을 해석하고 이해할 수 있습니다.
