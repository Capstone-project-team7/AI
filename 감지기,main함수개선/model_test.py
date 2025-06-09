# model_test.py - 분류 모델 테스트용 간소화 버전
import cv2
import asyncio
import logging
import time
import collections
import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# utils_api 임포트 추가 (API 연결용)
from utils_api import send_detection_info_to_server, format_detection_for_api

# 외부 파일에서 함수 가져오기
from advanced_theft_detection_model import theft_detection_model

# FastAPI 앱 생성
app = FastAPI(title="AI 모델 테스트 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 활성 비디오 프로세서 저장소
active_processors = {}

# 로깅 설정 개선
# 반복 경고 필터 클래스 정의
class DuplicateFilter(logging.Filter):
    def __init__(self, max_count=5, reset_interval=300):
        super().__init__()
        self.max_count = max_count
        self.reset_interval = reset_interval  # 초 단위로 카운터 리셋
        self.last_reset = time.time()
        self.msg_count = {}
        
    def filter(self, record):
        # 주기적으로 카운터 초기화
        current_time = time.time()
        if current_time - self.last_reset > self.reset_interval:
            self.msg_count = {}
            self.last_reset = current_time
            
        # 메시지 해시 생성 (로깅 레벨과 메시지 내용 기반)
        msg_hash = f"{record.levelname}:{record.getMessage()}"
        
        # 카운트 증가
        if msg_hash in self.msg_count:
            self.msg_count[msg_hash] += 1
        else:
            self.msg_count[msg_hash] = 1
            
        # 최대 반복 카운트 초과시 필터링
        if self.msg_count[msg_hash] > self.max_count:
            # 10배수마다 한 번씩만 로그 출력 (반복 상태 알림용)
            if self.msg_count[msg_hash] % (self.max_count * 10) == 0:
                record.getMessage = lambda: f"{record.getMessage()} (반복 {self.msg_count[msg_hash]}회)"
                return True
            return False
        return True

# 로그 설정
LOG_LEVEL = logging.INFO  # 테스트용으로 INFO 레벨로 변경
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 루트 로거에 중복 필터 추가
root_logger = logging.getLogger()
duplicate_filter = DuplicateFilter(max_count=3)  # 최대 3번까지만 같은 로그 허용
root_logger.addFilter(duplicate_filter)

# 메인 로거
logger = logging.getLogger(__name__)

# --- Configuration ---
SAVE_DIR = "recordings" # 영상 저장 디렉토리
BUFFER_SIZE = 300 # 버퍼 프레임 수 (예: 30fps 기준 10초)
RECORD_AFTER_DETECTION_FRAMES = 1800 # 감지 후 추가 녹화 프레임 (예: 30fps 기준 60초)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 중복 키포인트 경고 로그 필터링 함수 ---
filtered_keypoint_warnings = set()
last_keypoint_warning = 0
def filter_keypoint_warning(message):
    """키포인트 경고 로그를 필터링하여 동일한 경고는 한 번만 표시"""
    global filtered_keypoint_warnings, last_keypoint_warning
    current_time = time.time()
    
    # 5분마다 필터 초기화
    if current_time - last_keypoint_warning > 300:
        filtered_keypoint_warnings.clear()
        last_keypoint_warning = current_time
        
    if message in filtered_keypoint_warnings:
        return False
    
    filtered_keypoint_warnings.add(message)
    return True

# --- VideoProcessor Class ---
class VideoProcessor:
    def __init__(self, rtsp_url, cctv_id, user_id=None):
        self.rtsp_url = rtsp_url
        self.cctv_id = cctv_id
        self.user_id = user_id if user_id is not None else 1  # 기본 사용자 ID
        self.cap = None
        self.is_running = False
        self.curr_video_writer = None
        self.curr_video_path = None
        self.curr_video_start_time = None
        self.reconnect_interval = 5  # 재연결 시도 간격 (초)
        self.recording_dir = "recordings"
        self.frame_buffer = collections.deque(maxlen=30)  # 30프레임 버퍼 (약 1초)
        self.keypoints_buffer = collections.deque(maxlen=60)  # 포즈 키포인트 버퍼 (약 2초)
        self.last_detection_time = 0
        self.continuous_detection_count = 0
        self.detection_cooldown = 60  # 60초 쿨다운
        self.warning_counts = {}  # 각 경고 메시지의 카운터
        # 최근 감지된 행동 유형과 신뢰도 저장
        self.last_behavior_type = None
        self.last_confidence = 0.0

        # 테스트 모드에서는 록화 정보도 저장
        self.detections = []

        # 녹화 디렉토리 생성
        os.makedirs(self.recording_dir, exist_ok=True)
        
        # 녹화 상태 플래그
        self.is_recording = False

    async def start(self):
        logger.info(f"[{self.cctv_id}] 스트림 처리 시작: {self.rtsp_url}")
        self.is_running = True
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            logger.error(f"[{self.cctv_id}] RTSP 스트림에 연결할 수 없습니다: {self.rtsp_url}")
            return
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"[{self.cctv_id}] 비디오 정보: {width}x{height}, {fps}fps")
        
        try:
            await self.process_frames()
        except Exception as e:
            logger.error(f"[{self.cctv_id}] 프레임 처리 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.stop()

    async def start_recording(self):
        """새 비디오 녹화 시작"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.cctv_id}_{timestamp}.mp4"
        filepath = os.path.join(self.recording_dir, filename)
        
        # 첫 프레임의 해상도와 FPS 가져오기
        if self.frame_buffer:
            first_frame = self.frame_buffer[0]
            height, width = first_frame.shape[:2]
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:  # FPS를 가져올 수 없으면 기본값 사용
                fps = 30
                
            # H.264 계열 코덱 사용 (브라우저 호환성 최대화)
            codec_success = False
            codecs_to_try = [
                'avc1',     # 기본 H.264 코덱
                'h264',     # 일반 H.264
                'mp4v'      # 마지막 대안
            ]
            
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    self.curr_video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                    
                    if self.curr_video_writer.isOpened():
                        logger.info(f"[{self.cctv_id}] 🎥 녹화 시작: {os.path.basename(filepath)} (코덱: {codec})")
                        self.curr_video_path = filepath
                        self.curr_video_start_time = time.time()
                        self.is_recording = True
                        codec_success = True
                        break
                    else:
                        logger.warning(f"[{self.cctv_id}] {codec} 코덱으로 VideoWriter를 열 수 없음")
                        self.curr_video_writer = None
                except Exception as e:
                    logger.warning(f"[{self.cctv_id}] {codec} 코덱 사용 실패: {e}")
            
            if not codec_success:
                logger.error(f"[{self.cctv_id}] 어떤 코덱으로도 VideoWriter를 열 수 없습니다.")
                self.curr_video_writer = None
                self.curr_video_path = None
                self.is_recording = False
            
            # 버퍼에 있는 모든 프레임을 녹화 (이벤트 발생 전 영상 포함)
            if self.curr_video_writer is not None:
                for buffered_frame in self.frame_buffer:
                    self.curr_video_writer.write(buffered_frame)

    async def stop_recording(self):
        """현재 비디오 녹화 종료 및 저장"""
        if self.curr_video_writer is not None:
            self.curr_video_writer.release()
            logger.info(f"[{self.cctv_id}] 🛑 녹화 종료: {os.path.basename(self.curr_video_path)}")
            
            self.is_recording = False
            
            # 녹화 종료 후 API 서버로 알림 전송
            if self.last_behavior_type is not None:
                try:
                    video_url = f"file://{self.curr_video_path}"  # S3 없이 로컬 파일 경로로 대체
                    
                    # API 서버로 감지 정보 전송
                    detection_data = format_detection_for_api(
                        cctv_id=self.cctv_id,
                        videoUrl=video_url,
                        anomalyType=self.last_behavior_type,
                        confidence=self.last_confidence,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        thumbnail_url=None  # 썸네일 없음
                    )
                    
                    # API 호출
                    success, response = await send_detection_info_to_server(detection_data)
                    
                    if success:
                        logger.info(f"[{self.cctv_id}] 감지 정보 전송 성공: {response}")
                    else:
                        logger.error(f"[{self.cctv_id}] 감지 정보 전송 실패: {response}")
                except Exception as e:
                    logger.error(f"[{self.cctv_id}] API 호출 중 오류: {e}")
            
            self.curr_video_writer = None
            self.curr_video_path = None
            self.curr_video_start_time = None

    async def process_frames(self):
        frame_count = 0
        last_warning_time = 0
        max_warnings_per_minute = 3  # 분당 최대 경고 메시지 수
        
        while self.is_running:
            success, frame = self.cap.read()
            
            if not success:
                current_time = time.time()
                # 연결 재시도 경고를 1분에 최대 3번만 표시
                if current_time - last_warning_time > 60 / max_warnings_per_minute:
                    logger.warning(f"[{self.cctv_id}] 프레임을 읽을 수 없음. 재연결 시도...")
                    last_warning_time = current_time
                await asyncio.sleep(self.reconnect_interval)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue
            
            # 프레임 버퍼에 추가
            self.frame_buffer.append(frame.copy())
            
            # 현재 녹화 중이면 프레임 저장
            if self.curr_video_writer is not None:
                self.curr_video_writer.write(frame)
            
            # 10 프레임마다 절도 감지 (성능 최적화)
            if frame_count % 10 == 0:
                # 이상행동 감지 모델 실행
                try:
                    detection_result = theft_detection_model(frame, self.keypoints_buffer)
                    
                    if detection_result and isinstance(detection_result, tuple) and len(detection_result) >= 3:
                        is_anomaly, confidence, behavior_type = detection_result
                        
                        if is_anomaly:
                            current_time = time.time()
                            
                            # 쿨다운 시간이 지났으면 새로운 감지로 처리
                            if current_time - self.last_detection_time > self.detection_cooldown:
                                self.continuous_detection_count = 1
                            else:
                                self.continuous_detection_count += 1
                            
                            self.last_detection_time = current_time
                            
                            # 연속 감지 횟수가 1회 이상이면 실제 이벤트로 간주
                            if self.continuous_detection_count >= 1:
                                logger.warning(f"[{self.cctv_id}] 🚨 이상행동 감지: {behavior_type} (신뢰도: {confidence:.2f})")
                                
                                # 최근 감지된 행동 유형과 신뢰도 저장
                                self.last_behavior_type = behavior_type
                                self.last_confidence = confidence
                                
                                # 탐지 내역 저장 (테스트용)
                                self.detections.append({
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "type": behavior_type,
                                    "confidence": confidence
                                })
                                
                                # 녹화 중이 아니면 녹화 시작
                                if not self.is_recording:
                                    await self.start_recording()
                except Exception as e:
                    # 키포인트 추출 경고는 필터링
                    error_msg = str(e)
                    if "추출된 키포인트의 형태가 비정상" in error_msg:
                        # 이 경고는 너무 많이 발생하므로 특별히 필터링
                        if error_msg not in self.warning_counts:
                            self.warning_counts[error_msg] = 0
                        self.warning_counts[error_msg] += 1
                        
                        # 처음 5번만 로그, 이후 100번마다 한 번씩만 로그
                        if self.warning_counts[error_msg] <= 5 or self.warning_counts[error_msg] % 100 == 0:
                            if self.warning_counts[error_msg] > 5:
                                logger.warning(f"[{self.cctv_id}] {error_msg} (발생 횟수: {self.warning_counts[error_msg]})")
                            else:
                                logger.warning(f"[{self.cctv_id}] {error_msg}")
                    else:
                        logger.error(f"[{self.cctv_id}] 감지 모델 실행 중 오류: {str(e)}")
                
            # 녹화 중이고 마지막 감지 이후 일정 시간이 지났으면 녹화 종료
            if self.is_recording and current_time - self.last_detection_time > 10:  # 10초 후 종료
                await self.stop_recording()
                
            frame_count += 1
            await asyncio.sleep(0.001)  # 다른 작업에 CPU 시간 양보

    def stop(self):
        """비디오 처리 종료"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            
        if self.curr_video_writer is not None:
            self.curr_video_writer.release()
            
        logger.info(f"[{self.cctv_id}] 스트림 처리 종료")
        
        # 테스트 결과 출력
        if self.detections:
            logger.info(f"[{self.cctv_id}] 감지된 이상행동 총 {len(self.detections)}건:")
            for i, detection in enumerate(self.detections):
                logger.info(f"  {i+1}. {detection['timestamp']} - {detection['type']} (신뢰도: {detection['confidence']:.2f})")
        else:
            logger.info(f"[{self.cctv_id}] 감지된 이상행동 없음")

# --- API 엔드포인트 ---

@app.post("/api/v1/streaming/start")
async def start_stream(payload: dict = Body(..., example={"cctv_id": 456, "user_id": 123, "rtsp_url": "rtsp://..."})):
    """새로운 RTSP 스트림 모니터링 시작"""
    cctv_id_raw = payload.get("cctv_id")
    
    # cctv_id 숫자 처리
    try:
        # 숫자로 변환 시도
        if isinstance(cctv_id_raw, (int, float)):
            cctv_id = int(cctv_id_raw)
        elif isinstance(cctv_id_raw, str) and cctv_id_raw.strip().isdigit():
            cctv_id = int(cctv_id_raw)
        else:
            # 변환 불가능한 경우 (예외적인 경우)
            cctv_id = cctv_id_raw
    except Exception:
        # 변환 실패 시 원래 값 사용
        cctv_id = cctv_id_raw
    
    user_id = payload.get("user_id")
    rtsp_url = payload.get("rtsp_url")
    
    if not cctv_id or not rtsp_url:
        raise HTTPException(status_code=400, detail="cctv_id와 rtsp_url이 필요합니다")
    
    # 문자열 키로 딕셔너리 처리를 위해 키를 문자열로 변환
    cctv_id_key = str(cctv_id)
    
    # 이미 실행 중인 스트림인지 확인
    if cctv_id_key in active_processors:
        # 기존 프로세서 중지
        active_processors[cctv_id_key].stop()
        logger.warning(f"기존 스트림 '{cctv_id}' 중지됨")
    
    # 새 비디오 프로세서 생성 및 시작
    processor = VideoProcessor(rtsp_url, cctv_id, user_id)
    active_processors[cctv_id_key] = processor
    
    # 비동기 작업으로 실행
    asyncio.create_task(processor.start())
    
    logger.warning(f"✅ 스트림 '{cctv_id}' 시작됨: {rtsp_url}")
    return {"status": "success", "message": f"스트림 '{cctv_id}' 시작됨"}

@app.put("/api/v1/streaming/stop")
async def stop_stream_by_user(payload: dict = Body(..., example={"user_id": 1, "cctv_id": 456})):
    """Spring에서 검증된 user_id와 cctv_id 기반으로 스트림 중지"""
    user_id = payload.get("user_id")
    cctv_id_raw = payload.get("cctv_id")
    
    # cctv_id 숫자 처리
    try:
        if isinstance(cctv_id_raw, (int, float)):
            cctv_id = int(cctv_id_raw)
        elif isinstance(cctv_id_raw, str) and cctv_id_raw.strip().isdigit():
            cctv_id = int(cctv_id_raw)
        else:
            cctv_id = cctv_id_raw
    except Exception:
        cctv_id = cctv_id_raw
    
    cctv_id_key = str(cctv_id)

    if not user_id or not cctv_id:
        raise HTTPException(status_code=400, detail="user_id와 cctv_id가 필요합니다")

    if cctv_id_key not in active_processors:
        raise HTTPException(status_code=404, detail=f"스트림 '{cctv_id}'를 찾을 수 없습니다")

    processor = active_processors[cctv_id_key]
    if str(processor.user_id) != str(user_id):
        raise HTTPException(status_code=403, detail="이 스트림을 중지할 권한이 없습니다")

    processor.stop()
    del active_processors[cctv_id_key]

    logger.warning(f"🔴 스트림 '{cctv_id}' (user {user_id}) 중지됨")
    return {"status": "success", "message": f"스트림 '{cctv_id}' 중지됨"}

@app.get("/api/v1/streaming/list")
async def list_streams():
    """현재 실행 중인 모든 스트림 목록 조회"""
    streams = []
    for cctv_id_key, processor in active_processors.items():
        streams.append({
            "cctv_id": processor.cctv_id,
            "user_id": processor.user_id,
            "rtsp_url": processor.rtsp_url,
            "is_recording": processor.is_recording,
            "detections_count": len(processor.detections)
        })
    return {"status": "success", "streams": streams}

@app.get("/api/v1/test")
async def test_model_api():
    """AI 모델 테스트 API"""
    result = await test_model()
    return result

# --- 테스트 실행 함수 ---
async def run_test(rtsp_url, test_duration=60, cctv_id=1):
    """
    지정된 시간 동안 RTSP 영상에서 이상행동 감지 테스트 실행
    
    Args:
        rtsp_url: RTSP 스트림 URL 또는 비디오 파일 경로
        test_duration: 테스트 실행 시간 (초)
        cctv_id: 테스트용 CCTV ID
    """
    logger.info(f"이상행동 감지 테스트 시작: {rtsp_url} (테스트 시간: {test_duration}초)")
    
    processor = VideoProcessor(rtsp_url, cctv_id)
    process_task = asyncio.create_task(processor.start())
    
    try:
        # 지정된 시간 동안 실행
        await asyncio.sleep(test_duration)
        
        # 테스트 종료
        processor.is_running = False
        await process_task
    except asyncio.CancelledError:
        processor.stop()
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        processor.stop()
    
    logger.info("이상행동 감지 테스트 종료")

# --- 이미지 파일로 테스트 ---
async def test_image(image_path):
    """
    이미지 파일로 모델 테스트
    
    Args:
        image_path: 이미지 파일 경로
    """
    if not os.path.exists(image_path):
        logger.error(f"이미지 파일을 찾을 수 없음: {image_path}")
        return
    
    logger.info(f"이미지 테스트 시작: {image_path}")
    
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 로드할 수 없음: {image_path}")
            return
        
        # 키포인트 버퍼 생성
        buffer = collections.deque(maxlen=60)
        
        # 모델로 이미지 분석
        result = theft_detection_model(image, buffer)
        
        if result and isinstance(result, tuple) and len(result) >= 3:
            is_anomaly, confidence, behavior_type = result
            
            if is_anomaly:
                logger.info(f"🚨 이상행동 감지: {behavior_type} (신뢰도: {confidence:.2f})")
            else:
                logger.info(f"정상 행동 (신뢰도: {confidence:.2f})")
        else:
            logger.info("행동 감지 결과 없음")
            
    except Exception as e:
        logger.error(f"이미지 테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    logger.info("이미지 테스트 종료")

# --- 비디오 파일로 테스트 ---
async def test_video(video_path, duration=None):
    """
    비디오 파일로 모델 테스트
    
    Args:
        video_path: 비디오 파일 경로
        duration: 테스트 지속 시간 (초), None이면 전체 비디오
    """
    if not os.path.exists(video_path):
        logger.error(f"비디오 파일을 찾을 수 없음: {video_path}")
        return
    
    logger.info(f"비디오 테스트 시작: {video_path}")
    await run_test(video_path, test_duration=duration or 9999, cctv_id="test")
    logger.info("비디오 테스트 종료")

# --- 모델 테스트 함수 ---
async def test_model():
    """
    모델 로딩 및 기본 테스트 실행
    """
    logger.info("모델 테스트 시작")
    try:
        # 테스트 이미지 생성 (검은색 배경의 빈 이미지)
        test_image = np.zeros((640, 480, 3), dtype=np.uint8)
        
        # 빈 keypoints_buffer 생성
        test_keypoints_buffer = collections.deque(maxlen=60)
        
        # 모델 테스트 실행
        result = theft_detection_model(test_image, test_keypoints_buffer)
        
        # 결과 분석
        if result and isinstance(result, tuple) and len(result) >= 3:
            is_anomaly, confidence, behavior_type = result
            
            logger.info(f"모델 테스트 결과: is_anomaly={is_anomaly}, confidence={confidence:.2f}, behavior_type={behavior_type}")
            return {
                "status": "success",
                "message": "AI 모델 테스트 성공",
                "result": {
                    "is_anomaly": is_anomaly,
                    "confidence": float(confidence),
                    "behavior_type": behavior_type if behavior_type else "None"
                }
            }
        else:
            logger.info("모델 테스트 결과 없음")
            return {
                "status": "success",
                "message": "AI 모델 테스트 성공 (결과 없음)",
                "result": {
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "behavior_type": "None"
                }
            }
    except Exception as e:
        logger.error(f"AI 모델 테스트 중 오류 발생: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        return {
            "status": "error",
            "message": f"AI 모델 테스트 중 오류 발생: {e}",
            "traceback": error_traceback
        }

# 테스트 실행 함수
async def main():
    """
    메인 테스트 함수
    """
    logger.info("이상행동 감지 모델 테스트 시작")
    
    # 1. 모델 로딩 테스트
    model_test_result = await test_model()
    logger.info(f"모델 로딩 테스트 결과: {model_test_result['status']}")
    
    # API 서버 시작 메시지
    logger.info("모델 테스트 API 서버가 시작되었습니다.")
    logger.info("다음 API 엔드포인트가 사용 가능합니다:")
    logger.info("  - POST /api/v1/streaming/start: 스트림 시작")
    logger.info("  - PUT /api/v1/streaming/stop: 스트림 종료")
    logger.info("  - GET /api/v1/streaming/list: 스트림 목록")
    logger.info("  - GET /api/v1/test: 모델 테스트")
    logger.info("Ctrl+C를 눌러 서버 종료")

# 테스트 메인 실행
if __name__ == "__main__":
    # 필요한 디렉토리가 없으면 생성
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("test_videos", exist_ok=True)
    
    # 모델 로딩 테스트 및 API 서버 실행
    asyncio.run(main())
    
    # FastAPI 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8008) 