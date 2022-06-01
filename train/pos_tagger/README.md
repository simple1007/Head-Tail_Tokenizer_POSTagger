# 품사태거

## 학습데이터 다운로드(형태소 분석 데이터, 품사 태기 데이터 1대1 매칭됨)
* Head-Tail 형태소 분석 데이터
  * [다운로드](https://drive.google.com/file/d/10OofteydOIgJbd0FMrLOrviMi809wD3o/view?usp=sharing)
* Head-Tail 형태소 품사태깅 데이터
  * [다운로드](https://drive.google.com/file/d/1EzXt0q64edj6xww8y3jGVVuh-btbbTZa/view?usp=sharing)

## 품사태거 데이터셋 Numpy File Create
* 기존에 사용한 Numpy 학습 데이터
  * [다운로드](https://drive.google.com/file/d/1jLkQOcSCQifmoXfyCG6zPqiYtRFGpvOJ/view?usp=sharing)
* Numpy 학습 데이터 생성
  * python data_numline_all_tag.py
  ```c 
  usage: POS Tagger Numpy Data Create [-h] [--MAX_LEN MAX_LEN] [--BATCH BATCH] [--input_morph INPUT_MORPH]
                                    [--input_tag INPUT_TAG] [--stop_batch_num STOP_BATCH_NUM]

  optional arguments:
    -h, --help            show this help message and exit
    --MAX_LEN MAX_LEN     MAX Sequnce Length ( 버트 입력 시퀀스 길이:BERT Input Tokenizer Length )
    --BATCH BATCH         BATCH Size
    --input_morph INPUT_MORPH
                          Input line morph file path (학습할 Head-Tail 형태소 분석 파일 경로) 
    --input_tag INPUT_TAG
                          Input line tag file path (학습할 Head-Tail 형태소 품사태깅 파일 경로)
    --stop_batch_num STOP_BATCH_NUM
                          Make Stop Batch Num (생성할 데이터셋 배치 개수)
  ```
* Default Arguments
  * --MAX_LEN 200
  * --BATCH 50
  * --input_morph ./kcc150_morphs.txt
  * --input_tag" ./kcc150_tag.txt
  * --stop_batch_num 8000
  
## Head-Tail 품사태거 학습
* python train_tkbigram_pos_tagger.py
```c
usage: train_tkbigram_pos_tagger.py [-h] [--MAX_LEN MAX_LEN] [--BATCH BATCH] [--EPOCH EPOCH] [--epoch_step EPOCH_STEP]
                                    [--validation_step VALIDATION_STEP] [--hidden_state HIDDEN_STATE]
                                    [--GPU_NUM GPU_NUM] [--model_name MODEL_NAME]

Postagger

optional arguments:
  -h, --help            show this help message and exit
  --MAX_LEN MAX_LEN     MAX Sequnce Length  (품사태거 데이터셋 Numpy File Create 에서 설정한 MAX_LEN 값)
  --BATCH BATCH         BATCH Size
  --EPOCH EPOCH         EPOCH Size
  --epoch_step EPOCH_STEP
                        Train Data Epoch Step (학습할 BATCH의 개수)
  --validation_step VALIDATION_STEP
                        Validation Data Epoch Step (검증할 BATCH의 개수)
  --hidden_state HIDDEN_STATE
                        BiLstm Hidden State (BiLSTM 출력층 차원 설정)
  --GPU_NUM GPU_NUM     Train GPU NUM (사용할 GPU Number)
  --model_name MODEL_NAME 
                        Tokenizer Model Name (저장할 Model Name)
```
* Default Arguments
  * --MAX_LEN 200
  * --BATCH 50
  * --EPOCH 5
  * --epoch_step 4000
  * --validation_step 240
  * --hidden_state tag_len*2
  * --GPU_NUM 0
  * --model_name tkbigram_one_first_alltag_bert_tagger.model
