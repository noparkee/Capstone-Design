# Capstone-Design
2021 COSE489

### 1. Data Augmentation   
지폐와 동전 앞/뒷면 각각 10장씩 가지고 있는 데이터를 증강.   
- 지폐의 경우 5가지 범위로 crop, 45도 단위로 rotate, 블러   
- 동전의 경우 가운데 부분만 crop, 45도 단위로 rotate, 블러   

### 2. Classifier
Conv2D를 이용한 단순한 classiier 모델   
현재 성능은 90% 정도 (211014)
