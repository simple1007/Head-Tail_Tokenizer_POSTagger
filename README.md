# Head-Tail_Tokenizer_POSTagger
어휘 형태소 Head 문법 형태소 Tail 두 개의 형태소 단위로만 분리하는 Head-Tail 토크나이저 품사태거

* /bin Head-Tail 실행
* /train Head-Tail 학습

사용목적
> 1. Head-Tail 분석후 명사만을 추출하여 사용
> 2. 한국어의 용언에서 어휘형태소의 원형 변형이 일어나지만 용언의 원형 변형으로 인한 토큰의 증가는 미미함
> 3. 모든 어휘를 사용 한다면 문법 형태소의 부분인 Tail은 버리고 사용할 것
> 4. BERT 등 Subword Tokenizer를 사용하는 모델에서는 MRC, NER, 군문오류 탐지 등 문서내에서 답을 찾거나, 태깅등의 태스크에서는 모델이 예측시 불필요한 문법형태소가 같이 추출됨 
> > Head-Tail 분석후 -> Subword Tokenizer를 사용시 성능향상 (딥러닝 모델에 적합한 토큰화 방식)
