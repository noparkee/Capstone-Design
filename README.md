# Capstone-Design
2021 COSE489   

_주어진 이미지 인풋에 대해 화폐인지 아닌지 분류하고, 화폐인 경우 어떤 종류의 화폐인지 분류한다._

---

## train_model.py
모델 학습하는 코드, 학습 후 SavedModel 타입과, tflite 타입으로 각각 저장   

### 화폐인지 아닌지를 분류 하는 모델 학습
```console
python train_model.py --type binary
```

### 화폐를 분류하는 모델 학습
```console
python train_model.py
```

SavedModel: [연도-월-일-시-분]\_[모델타입]\_[EPOCHES]\_[BATCH_SIZE]\_[IMG_HEIGTH]x[IMG_WIDTH].  
tflite: [연도-월-일-시-분]\_[모델타입]\_[EPOCHES]\_[BATCH_SIZE]\_[IMG_HEIGTH]x[IMG_WIDTH].tflite
2가지 저장


## test_model.py
SavedModel 타입의 모델을 로드해서 테스트   
```console
python test_model.py --binary-model [binary_model_path] --classifier-model [classifier_model] --file-path [test_file_path]
```

## test_tflite_model.py
tflite 타입의 모델을 로드해서 테스트   
```console
python test_tflite_model.py --binary-model [binary_model_path] --classifier-model [classifier_model] --file-path [test_file_path]
```
