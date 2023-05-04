# Head-Tail 분석

## Head-Tail Model File 다운로드 후 해당폴더에 압축해제
* 모델 파일 다운로드
[링크](https://drive.google.com/file/d/1MKdsrMn0smQVtaG-cL_eM2NVCaum3R9k/view?usp=sharing)
* distil kobert 품사태거
[링크](https://drive.google.com/file/d/1oA0mxv8tNFWQkEI8xgPkt98Tj1ISV1cj/view?usp=sharing)
## Head-Tail 분석
* python head-tail.py 실행
* 또는 python head_tail_distil.py 살행(distil kobert postagger)
```c 
# 입력 문자 분석
mode file!!"filename",or text!!"text" or exit:text!!나는 밥을 먹고 학교에 갔다.
"분석 결과 출력"

# 입력 파일 분석
mode file!!"filename",or text!!"text" or exit:file!!test.txt
output file name:result.txt(분석결과 저장할 파일명 입력)

# 종료
mode file!!"filename",or text!!"text" or exit:exit
```
