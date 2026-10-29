# DMS PROJECT
임베디드 장치로 차량 내부에 탑승한 탑승자 수를 확인하고, 운전자를 식별하여 운전자의 특성과 졸음여부 주시상태를 감시하는 Driver Management System 개발 입니다.

## 구현 목표 설정
- 등록된 운전자 식별 및 미등록 운전자 구분
- 눈 감김과 PERCLOS 기반 졸음 감지
- 머리 자세 및 시선 이탈 감지
- ROS2 기반 위험도 판단과 경고 출력
- 카메라·모델 오류 감시 및 개인정보 보호

## 환경 설정
```
sudo apt update
sudo apt install \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-diagnostic-updater \
  python3-colcon-common-extensions \
  python3-rosdep
```
```
python -m pip install --upgrade pip
python -m pip install \
	mediapipe \
	numpy==1.26.4 \
	opencv-python==4.10.0 \
```

## ROS2 패키지 설명
| 패키지 | 담당 기능 | 주요 출력 |
| :--- | :--- | :--- |
| dms_msgs | 공통 메시지·서비스 정의 | 메시지 인터페이스 |
| dms_core | 운전자 식별, 졸음·시선·고개 자세 분석, 위험 판단 | 운전자 상태 |
| dms_hardware | 카메라 입력 및 경고 출력 | 영상 및 진단 |
| dms_bringup | 노드 실행, 파라미터 전달, 전체 시스템 구성 | 통합 실행 |

## 참조 알고리즘
- 얼굴 검출: OpenCV Zoo YuNet https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- 졸음 감지: MediaPipe Face Landmarker https://developers.google.cn/edge/mediapipe/solutions/vision/face_landmarker/python
- 주시 상태 분석: OpenVINO 모델 https://docs.openvino.ai/2023.3/omz_demos_gaze_estimation_demo_cpp.html
- 운전자 식별: InsightFace Model Zoo https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
