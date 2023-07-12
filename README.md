# Head-Tail_Tokenizer_POSTagger
어휘 형태소 Head 문법 형태소 Tail 두 개의 형태소 단위로만 분리하는 Head-Tail 토크나이저 품사태거

* /bin Head-Tail 실행
* /train Head-Tail 학습

사용목적
> 1. Head-Tail 분석후 명사만을 추출하여 사용 
> > * !복합명사 추출 기능은 없음 -> 그로 인한 토큰의 다양성 증가가 있음을 고려 바랍니다.
> > * 한국어의 용언에서 어휘형태소의 원형 변형이 일어나지만 용언의 원형 변형으로 인한, 용언 Head 토큰의 개수 증가는 미미함 또한 대부분은 명사만을 사용하는 경우가 다수임
> 2. 모든 어휘를 사용 한다면 문법 형태소의 부분인 Tail은 버리고 사용할 것
> 3. BERT 등 Subword Tokenizer를 사용하는 모델에서는 MRC, NER, 구문오류/특정 단어 탐지 등 문서내에서 답을 찾거나, 태깅등의 태스크에서는 모델이 예측시 불필요한 문법형태소가 같이 추출됨 
> > Head-Tail 분석후 -> Subword Tokenizer를 사용시 성능향상 (원문형태를 보존 -> 딥러닝 모델에 적합한 토큰화 방식)

실험
# Head-Tail 토큰화 실험
## Classification
* 명사 추출후 Hate Speech Class 분류
* 복합 명사 미분해로 인한 성능저하 있음 추후 적용하여 실험 예정

| Tokenizer / Score Matrix | Accuracy | F1-Score |
| :---:   | :---: | :---: |
| okt | 83.67 | 87.59 |
| hannanum | 82.91 | 87.03 |
| komoran | 88.91 | 91.64 |
| mecab | 88.26 | 91.09 |
| **Head-Tail** | 86.13 | 89.59 |

* Ngram 언어모델을 이용한 자동띄어쓰기로 띄어쓰기 오류 수정
* 어미, 조사, 부호 토큰을 제외한 모든 토큰 사용
* NSMC 네이버 영화 리뷰 분류
  
| Tokenizer / Score Matrix | Accuracy | F1-Score |
| :---:   | :---: | :---: |
| okt | 82.88 | 82.75 |
| hannanum |  |  |
| komoran | 82.15 | 81.76 |
| mecab |  |  |
| kkma | 81.76 | 80.85 |
| **Head-Tail**(Komoran 복합명사 분해: 분석결과에 명사가 아닌것이 있음 분해안함) | 82.51 | 82.69 |
| **Head-Tail**(복합명사 분해안함) | 81.59 | 81.01 |

## MRC
* Distil KoBERT MRC(w/o Head-Tail, Head-Tail) [품사태그는 사용안함]
* 어휘형태소와 문법형태소가 분리되어 있어 문법형태소가 포함되지 않을 확률이 커진것으로 보임
* 모델 자체의 성능 보다는 정답 라벨의 범위를 깔끔하게 찾아내는것이 핵심
* Test Dataset 100,000 lines

| Tokenizer / Score Matrix | Rouge1-Score(recall/precision) | Accuracy | F1-Score |
| :---:   | :---: | :---: | :---: |
| w/o Head-Tail | 73.80/72.00 | 61.50 | 72.89 |
| **Head-Tail after Subword Tokenization** | 81.80/78.23 | 67.68 | 79.97 |
