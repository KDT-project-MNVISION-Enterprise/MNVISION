KDT 5기 혜모팀의 기업 프로젝트입니다.

# 엠엔비젼(MNVISION) : 작업자 안전관리 솔루션

![alt text](README_IMAGE/image.png)

## 소개

제조산업 안전에 대한 법령 강화와 중대 재해에 대한 엄격한 처벌 시행에 따라, 작업 공간 내에서의 안전사고 발생을 최소화하기 위한 노력이 진행되고 있습니다.

현재에는 라이트커튼과 지게차 작업 공간 라이트 센서를 사용하고 있지만,
사용자가 언제든지 임의로 끌 수 있기 때문에 단점이 존재합니다.

따라서, 현대자동차 울산공장 내에 설치된 CCTV 영상 데이터를 사용하여 내부 안전사고의 원인과 패턴을 파악하고 예방하기 위한 프로젝트를 수행하였습니다.

![alt text](README_IMAGE/image-2.png)

<hr>

# 결과물

#### 1. Video upload/manipulation

![alt text](README_IMAGE/video_manipulation.gif)

#### 2. Person-Rack

![alt text](README_IMAGE/Person-Rack.gif)

#### 3. Folklift-Person

![alt text](README_IMAGE/Folklift_Person.gif)

<hr>

### 개발환경

- 운영체제 : Windows 11
- 개발환경 : Visual Studio Code
- 프레임워크 : PyTorch, Flask
- 언어 : Python3.8
- 라벨링 프로그램 : LabelImg
- 패키지 : Ultralytics, PyQt5, Opencv, numpy, Pygame, csv, threading, time, os, re, datetime, subprocess, collections

## Stack

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![Tensorflow](https://img.shields.io/badge/Tensorflow-1877F2.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Qt](https://img.shields.io/badge/Qt-%23217346.svg?style=for-the-badge&logo=Qt&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Threads](https://img.shields.io/badge/Threads-000000?style=for-the-badge&logo=Threads&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

### 협력 업체

- 엠엔비젼(MNVISION) : [링크](http://mnvision.co.kr/)

<hr>

### 구성원

임소영(PM) : [Github](https://github.com/YimSoYoung1001)  
명노아 : [Github](https://github.com/noah2397)  
변주영 : [Github](https://github.com/rileybyun)  
손예림 : [Github](https://github.com/osllzd)  
이화은 : [Github](https://github.com/Skylee0310)

<hr>

## 일정

| 날짜       | 내용                                               | 링크                                                                                                                                        |
| ---------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024.05.01 | 팀 프로젝트 초기 설정                              | [링크](https://docs.google.com/document/d/11rDlDIC1FXRp6a_kL96bDIyI4A_ODqwx/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.02 | 프로젝트 관련 시각화 자료 조사                     | [링크](https://docs.google.com/document/d/1mukZDHGrvG5kVRodtRGQBNuXZJ940G_x/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.07 | 비젼 관련 기술 시장 조사                           | [링크](https://docs.google.com/document/d/1RtYCeMjceL8v2CEXiq8HHyaHXghZ9MqC/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.08 | 기업 미팅 진행, 요구사항 정의서 작성               | [링크](https://docs.google.com/document/d/1hnzKoSHDoXM1po03oT-R1nz527VQscft/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.10 | 프로젝트 계획 발표 & 기타 안건 발의                | [링크](https://docs.google.com/document/d/1myksuGxAPvzxuBnPrAHO1af6OlWAGEtG/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.13 | 데이터 전처리 관련 논의 및 세부 일정 조정          | [링크](https://docs.google.com/document/d/1rMVa2UPVCafioI9-BDuwem2E5fmj7ZXu/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.14 | 라벨링 기준 & 위험상황 설정, 개발파트 역할 분담 등 | [링크](https://docs.google.com/document/d/1xDl8Je9LnJwNPL_fEvt33dNVpl1Eg2Mf/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.16 | 기업 멘토링 진행 및 전처리 포멧 통일               | [링크](https://docs.google.com/document/d/1psknJ8tgaKesQXThGsKbDoi-pXdKP5ra/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.17 | 기술 구현 일정 수립                                | [링크](https://docs.google.com/document/d/1lS0y8dK35WL4l-HKOqXSyOfSOgaLhnwA/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.20 | 모델 & GUI 개발 진척 및 상황 공유                  | [링크](https://docs.google.com/document/d/1SGRziYjfBzMbeZEUnUh-xw76FwB4KXQ5/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.22 | 엠엔비젼 기업 멘토링 진행                          | [링크](https://docs.google.com/document/d/1aVqi9LIch3ZNc00td42e__VO7VneRFEn/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.24 | 데이터 라벨링 가이드 라인 정의                     | [링크](https://docs.google.com/document/d/1seDrVYlw7DjCM_P3ciyhf-i7vJj2hey4/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.29 | 엠엔비젼 기업 멘토링 진행                          | [링크](https://docs.google.com/document/d/1s9GWb8Pac_2y54SlmUtBslyjnl9yKBWs/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.05.30 | 중간발표 리허설 관련 회의                          | [링크](https://docs.google.com/document/d/12wdTpKA0ADbHRNzFEGknRAcyQXfhp_ah/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |
| 2024.06.05 | 기업 멘토링 진행                                   | [링크](https://docs.google.com/document/d/1f09FQL97ZL6NdreR_2KuWQslNfCN9xNx/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true) |

<hr>

### 기술 문서

1. 프로젝트 계획서(PPT) : [링크](https://www.canva.com/design/DAGEsOPGjEs/bs7mmdzugZokbFrcWfyEUA/view?utm_content=DAGEsOPGjEs&utm_campaign=designshare&utm_medium=link&utm_source=editor)
2. 간트 차트 : [링크](https://docs.google.com/spreadsheets/d/1jIBuAatHE040tEuWWt_ELjKuuFLut5WB/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true)
3. 일일업무보고 : [링크](https://docs.google.com/spreadsheets/d/1Hw6-rosnK7MumOWz8aodKlhbSH0-8ZpC/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true)
4. 요구사항 정의서 : [링크](https://docs.google.com/spreadsheets/d/1aT5VzIIrWiKlmHyt_mkaFU7nSRwweqj5/edit?usp=sharing&ouid=110067098172194561192&rtpof=true&sd=true)
5. 클래스 다이어그램 : [링크](https://drive.google.com/file/d/1fm_OE6rampXj4xIVq3pwiGv0Zh4FYyPU/view?usp=sharing)
6. 시퀀스 다이어그램 : [링크](https://drive.google.com/file/d/1zWUJ-RWFqSquwsO3cUs91bt_WegFkMkh/view?usp=sharing)
7. 플로우 차트 : [링크]()

<hr>

### Notion

![alt text](README_IMAGE/image-1.png)
Notion : [링크](https://water-maple-b1a.notion.site/_-_-43f06266c0f84a9ba1832b29022a6afd?pvs=4)
