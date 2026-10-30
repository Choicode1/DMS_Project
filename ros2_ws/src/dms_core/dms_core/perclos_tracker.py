# 시간 창 기반 PERCLOS 계산

from collections import deque
import time


class PerclosTracker:
    """
    시간 기반 PERCLOS 계산기

    - 기본 측정 구간: 60초
    - 눈 감김 시간: 실제 경과 시간으로 누적
    - 얼굴 미검출: 유효 관측에서 제외
    - 프레임 간격이 max_gap_sec보다 길면 해당 구간 제외
    """

    def __init__(
        self,
        window_sec=60.0,
        max_gap_sec=0.5,
        min_observed_sec=10.0,
    ):
        self.window_sec = float(window_sec)
        self.max_gap_sec = float(max_gap_sec)
        self.min_observed_sec = float(min_observed_sec)

        # (시작 시간, 종료 시간, 눈 감김 여부)
        self.intervals = deque()

        self.last_timestamp = None
        self.last_eye_closed = None

    def update(self, eye_closed, timestamp=None):
        """
        eye_closed:
            True  = 양쪽 눈 감김
            False = 양쪽 눈 뜸
            None  = 얼굴 미검출 또는 측정 불가

        timestamp:
            time.monotonic() 기준 초 단위
        """

        if timestamp is None:
            timestamp = time.monotonic()

        now = float(timestamp)

        # 이전 상태가 유지된 시간 계산
        if self.last_timestamp is not None:

            dt = now - self.last_timestamp

            if dt < 0:
                raise ValueError("Timestamp must be monotonic")

            # 비정상적으로 긴 프레임 공백은 측정에서 제외
            if (
                0 < dt <= self.max_gap_sec
                and self.last_eye_closed is not None
            ):
                self.intervals.append((
                    self.last_timestamp,
                    now,
                    self.last_eye_closed
                ))

        # 현재 시각 이전의 오래된 구간 제거
        cutoff = now - self.window_sec

        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()

        self.last_timestamp = now
        self.last_eye_closed = eye_closed

        return self.get_result(now)

    def get_result(self, now=None):

        if now is None:
            now = time.monotonic()

        now = float(now)
        cutoff = now - self.window_sec

        closed_duration = 0.0
        observed_duration = 0.0

        for start, end, closed in self.intervals:

            # 현재 측정 구간과 겹치는 부분만 계산
            overlap_start = max(start, cutoff)
            overlap_end = min(end, now)

            duration = overlap_end - overlap_start

            if duration <= 0:
                continue

            observed_duration += duration

            if closed:
                closed_duration += duration

        if observed_duration > 0:
            perclos = closed_duration / observed_duration
        else:
            perclos = 0.0

        coverage = min(
            observed_duration / self.window_sec,
            1.0
        )

        ready = observed_duration >= self.min_observed_sec

        return {
            'perclos': perclos,
            'perclos_percent': perclos * 100.0,
            'closed_duration': closed_duration,
            'observed_duration': observed_duration,
            'coverage': coverage,
            'ready': ready,
        }

    def reset(self):
        self.intervals.clear()
        self.last_timestamp = None
        self.last_eye_closed = None
