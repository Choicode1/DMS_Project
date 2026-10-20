# DMS PROJECT
임베디드 장치로 차량 내부에 탑승한 탑승자 수를 확인하고, 운전자를 식별하여 운전자의 특성과 졸음여부 주시상태를 감시하는 Driver Management System 개발 입니다.


## 참조 알고리즘
- 졸음 감지: MediaPipe Face Landmarker https://developers.google.cn/edge/mediapipe/solutions/vision/face_landmarker/python
- 주시 상태 분석: OpenVINO 모델 https://docs.openvino.ai/2023.3/omz_demos_gaze_estimation_demo_cpp.html
- 운전자 식별: InsightFace Model Zoo https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
