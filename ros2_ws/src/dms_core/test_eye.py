
import cv2
import time

from dms_core.eye_analyzer import EyeAnalyzer
from dms_core.perclos_tracker import PerclosTracker


def main():

    model_path = (
        'src/dms_core/models/face_landmarker.task'
    )

    cap = cv2.VideoCapture(0)

    analyzer = EyeAnalyzer(
        model_path=model_path,
        ear_threshold=0.21
    )

    tracker = PerclosTracker(
        window_sec=60.0,
        max_gap_sec=0.5,
        min_observed_sec=10.0
    )

    if not cap.isOpened():
        print("Camera open failed")
        analyzer.close()
        return

    last_log_time = 0.0

    try:
        while True:

            ret, frame = cap.read()
            now = time.monotonic()

            if not ret:
                tracker.update(None, now)
                continue

            result = analyzer.analyze(frame)

            if result is not None:

                perclos_result = tracker.update(
                    result['eye_closed'],
                    now
                )

                frame = analyzer.draw(frame, result)

                perclos = perclos_result['perclos_percent']
                coverage = perclos_result['coverage'] * 100

                status = (
                    f"PERCLOS: {perclos:.1f}%"
                    if perclos_result['ready']
                    else "PERCLOS: WARMING UP"
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Coverage: {coverage:.1f}%",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                if now - last_log_time >= 1.0:

                    print(
                        f"EAR: {result['avg_ear']:.3f} | "
                        f"Closed: {result['eye_closed']} | "
                        f"PERCLOS: {perclos:.1f}% | "
                        f"Coverage: {coverage:.1f}%"
                    )

                    last_log_time = now

            else:
                # 얼굴을 찾지 못한 시간은 PERCLOS에서 제외
                tracker.update(None, now)

                cv2.putText(
                    frame,
                    "FACE NOT DETECTED",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("DMS Eye + PERCLOS", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        analyzer.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
