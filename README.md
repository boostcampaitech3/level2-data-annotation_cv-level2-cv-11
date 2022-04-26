# 🎨 [Pstage] CV 11조 CanVas 

<img width="1000" alt="스크린샷 2022-04-26 오후 10 41 27" src="https://user-images.githubusercontent.com/63924704/165313281-099ecd44-ff34-4995-9645-702bf871f89e.png">

- 대회 기간 : 2022.04.14 ~ 2022.04.21
- 목적 : 이미지 내 글자 영역 검출(Text Detection)

## Overview
카메라로 카드 내 카드 번호 자동 인식, 차량 번호 자동 인식 등 **OCR (Optimal Character Recognition)** 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나이다. 본 대회에서는 OCR task 중 `글자 검출` 모델의 성능을 고도화하는 것을 목표로 한다. 

---

## Dataset 

- 기본 데이터셋: ICDAR17_Korean (ICDAR17-MLT 서브셋)
- 전체 이미지 개수
  - Train: 536
  - Test: 300

---

## 멤버
| 김영운 | 이승현 | 임서현 | 전성휴 | 허석용 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/Cronople) | [Github](https://github.com/sseunghyuns) | [Github](https://github.com/seohl16) | [Github](https://github.com/shhommychon) | [Github](https://github.com/HeoSeokYong)

- `김영운` : EDA, Annotation data converting(Polygon to Rectangle)
- `이승현` : EDA, 전수조사 환경 세팅, Fine-tuning 적용
- `임서현` : EDA, AiHub 오픈 데이터셋 적용
- `전성휴` : EDA, Augmentation 실험
- `허석용` : EDA, Validate 기능 개발

---

### 프로젝트 수행 결과 

<p align="center">
<img width="800" height=500 alt="2" src="https://user-images.githubusercontent.com/63924704/165317463-0476b036-4972-4192-905c-783561240c06.png">
</p>

**1. 데이터 전수 조사**
- 대회에서 제공한 Train data 536개 이미지와 annotation tool을 활용하여 직접 labeling한 1288개의 이미지에서 잘못 레이블링된 케이스가 발견되어 전수 조사를 진행 
- 전수 조사 결과 잘못 레이블링된 경우, 학습에서 제외하였다.
  - 대회 데이터 536개 중 16개, 추가로 제공받은 1288개 데이터 중 127개의 이미지에서 mis-labeling 발견하였고, 최종적으로 143개의 이미지를 제거하였다.

**2. Text Detection Open Dataset 조사**
- 300개의 평가 데이터에 비해 학습 데이터가 적어 추가 데이터 확보가 필수적이라고 판단하였고, 이를 위해 OCR 관련 오픈 데이터셋 조사 진행
- AiHub와 ICDAR19 데이터셋 선정 후 대회 데이터셋과 비슷하게 annotation 돠어 있는지 분석하였고, 선별된 이미지를 학습에 사용하기로 결정
- AiHub 데이터셋은 데이터 수가 많았지만 대회 목적에 부합하는 고품질 데이터가 적어서 학습 시 결과가 좋지 않아 최종적으로 제외

**3. Validate**
- Test 데이터셋 분포와 비슷하게 Validation 데이터를 구성하여 본 대회의 평가지표인 DetEval score가 valid-test 간 점수가 align하도록 세팅
- 에폭마다 Validation셋의 Mean loss, Classification loss, Angle loss, IoU loss를 확인하여 최적의 모델을 찾을 수 있도록 실험 환경을 구성

**4. Fine-Tuning 전략**
- 제공받은 학습 데이터는 한글 및 영어 글자 영역을 포함하는 536개의 이미지이다.
- 총 10,000개의 데이터(한글, 영어 외 여러 언어 포함)가 존재하는 ICDAR19를 활용하여 먼저 모델을 학습한 후, target dataset인 536개의 학습 데이터에 대해 fine-tuning 진행
- 10,000여개의 데이터로 학습된 모델로 target dataset에 대해 fine-tuning을 취하는 전략은 모델의 견고성을 더해주었고, 모델 성능이 크게 향상되었다.

**5. Hyperparameter Tuning**
- Fine-tuning과 augmentation으로 학습된 모델을 최적화하기 위하여 learning rate을 더욱 낮춰 학습하는 등 hyperparameter tuning 진행 
- 최종 리더보드 0.6752 점수 기록(8위/19)


<img width="1112" alt="스크린샷 2022-04-26 오후 11 55 00" src="https://user-images.githubusercontent.com/63924704/165329198-9e96f146-8c4c-489b-a37f-5af1ac3de256.png">

---
