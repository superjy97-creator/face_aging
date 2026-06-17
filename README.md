# Face Aging Age Group Classification Web App

얼굴 이미지 기반 나이대 분류 웹앱 프로젝트입니다.  
AI Hub 안면 에이징 이미지 데이터를 기반으로 EfficientNet-B0 모델을 학습하고,  
Streamlit 웹앱에서 웹캠으로 촬영한 얼굴 이미지를 분석하여 나이대를 예측합니다.

## 프로젝트 개요

이 프로젝트는 얼굴 이미지를 입력받아 다음 5개 나이대 중 하나로 분류합니다.

| 클래스 | 나이대 |
|---|---|
| 0 | 10대 이하 |
| 1 | 20대 |
| 2 | 30대 |
| 3 | 40대 |
| 4 | 50대 이상 |

## 사용 모델

최종 선택 모델은 EfficientNet-B0입니다.

MobileNetV2와 비교한 결과, EfficientNet-B0가 더 높은 검증 성능을 보여 최종 모델로 선정했습니다.

## 주요 기능

- 웹캠 이미지 촬영
- EfficientNet-B0 best 모델 기반 나이대 예측
- 클래스별 예측 확률 표시
- 학습 결과 시각화 페이지
- 프로젝트 용어 정리 페이지

## 프로젝트 구조

```text
face_aging/
├─ webapp/
│  ├─ app.py
│  └─ requirements.txt
├─ outputs/
│  ├─ csv/
│  ├─ figures/
│  └─ models/
│     └─ efficientnet_b0_age_past_5class_best.pth
├─ README.md
└─ .gitignore