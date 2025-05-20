# main.py
import cv2
import asyncio
from fastapi import FastAPI, HTTPException, Body
import logging
import time
import collections
import os
import random # dummy_anomaly_detector가 random 사용 시 필요
import torch # 모델 로드에 필요
import shutil # 로컬 파일 복사/이동에 사용 가능
import warnings
import logging.handlers
import numpy as np  # test_model 함수에 필요

from utils_s3 import upload_to_s3, test_s3_connection, print_aws_credentials_info, setup_bucket_cors, check_bucket_public_access
from utils_api import send_detection_info_to_server, format_detection_for_api

# --- 외부 파일에서 함수 가져오기 ---

from advanced_theft_detection_model import theft_detection_model

# --- 로깅 설정 개선 ---
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
LOG_LEVEL = logging.WARNING  # 기본 로그 레벨을 WARNING으로 설정
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 루트 로거에 중복 필터 추가
root_logger = logging.getLogger()
duplicate_filter = DuplicateFilter(max_count=3)  # 최대 3번까지만 같은 로그 허용
root_logger.addFilter(duplicate_filter)

# opencv 로그 억제 (opencv에서 발생하는 AVI 관련 경고는 ERROR 레벨 이상만 표시)
logging.getLogger('opencv').setLevel(logging.ERROR)

# 라이브러리들의 로그 레벨 조정
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.WARNING)

# utils_s3에서 발생하는 로그는 항상 표시
logging.getLogger('utils_s3').setLevel(logging.INFO)

# OpenCV 경고 억제 (저수준 C++ 경고)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"  # FFmpeg 디버그 비활성화

# 서버 로거
logger = logging.getLogger('theft_detection_server')
logger.setLevel(logging.INFO)

# --- Configuration ---
SAVE_DIR = "recordings" # 영상 저장 디렉토리
BUFFER_SIZE = 300 # 버퍼 프레임 수 (예: 30fps 기준 10초)
RECORD_AFTER_DETECTION_FRAMES = 1800 # 감지 후 추가 녹화 프레임 (예: 30fps 기준 60초)
os.makedirs(SAVE_DIR, exist_ok=True)

# 다른 라이브러리의 로깅 레벨 설정
logging.getLogger('ultralytics').setLevel(logging.ERROR)  # YOLO 로그 최소화
logging.getLogger('PIL').setLevel(logging.WARNING)        # PIL 로그 최소화
logging.getLogger('matplotlib').setLevel(logging.WARNING) # matplotlib 로그 최소화

# 메인 로깅 설정 (WARNING 레벨로 설정하여 INFO 레벨 로그 줄이기)
logging.basicConfig(
    level=logging.WARNING,  # WARNING 이상 레벨만 표시
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# S3 관련 로그는 항상 표시되도록 설정
logging.getLogger('utils_s3').setLevel(logging.INFO)

# --- S3 업로드 성공/실패 이벤트 로그 함수 ---
def log_s3_event(success: bool, filepath: str, s3_key: str, url: str = None, error: str = None):
    """S3 업로드 이벤트를 눈에 띄게 로깅"""
    if success:
        logger.warning(f"✅ S3 업로드 성공: {filepath} -> {s3_key}")
        logger.warning(f"🔗 URL: {url}")
    else:
        logger.error(f"❌ S3 업로드 실패: {filepath} -> {s3_key}")
        if error:
            logger.error(f"❗ 오류: {error}")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="무인점포 절도 감지 및 녹화 서비스",
    description="CCTV 영상에서 이상행동을 감지하고 녹화하는 API",
    version="1.0.0"
)

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

# --- Global Variables ---
# 활성 비디오 프로세서를 저장할 딕셔너리 (CCTV별 처리)
active_processors = {}

# --- VideoProcessor Class ---
class VideoProcessor:
    def __init__(self, rtsp_url, cctv_id, user_id=None):
        self.rtsp_url = rtsp_url
        
        # cctv_id를 정수로 변환 시도
        try:
            if isinstance(cctv_id, str) and cctv_id.isdigit():
                self.cctv_id = int(cctv_id)
            elif isinstance(cctv_id, (int, float)):
                self.cctv_id = int(cctv_id)
            else:
                self.cctv_id = cctv_id  # 변환 불가능한 경우 원래 값 유지
        except Exception:
            # 변환 중 오류가 발생하면 원래 값 사용
            self.cctv_id = cctv_id
            
        self.stream_id = self.cctv_id  # 기존 코드와의 호환성 유지
        self.user_id = user_id if user_id is not None else 1  # 기본값 1
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

        # 녹화 디렉토리 생성
        os.makedirs(self.recording_dir, exist_ok=True)

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
                            
                            # 연속 감지 횟수가 2회 이상이면 실제 이벤트로 간주
                            if self.continuous_detection_count >= 1:
                                logger.warning(f"[{self.cctv_id}] 🚨 이상행동 감지: {behavior_type} (신뢰도: {confidence:.2f})")
                                
                                # 최근 감지된 행동 유형과 신뢰도 저장
                                self.last_behavior_type = behavior_type
                                self.last_confidence = confidence
                                
                                # 이전 비디오가 없으면 새로 시작
                                if self.curr_video_writer is None:
                                    await self.start_recording()
                                
                                # 이미 녹화 중이면 타임스탬프 업데이트
                                self.curr_video_start_time = time.time()
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
                
            # 현재 녹화 중이라면 프레임 저장
            if self.curr_video_writer is not None:
                self.curr_video_writer.write(frame)
                
                # 마지막 감지 후 10초가 지나면 녹화 종료
                if time.time() - self.curr_video_start_time > 10:
                    await self.stop_recording()
            
            frame_count += 1
            await asyncio.sleep(0.001)  # 다른 작업에 CPU 시간 양보

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
            # ffmpeg 설치 후 사용 가능한 고급 코덱들 시도
            codecs_to_try = [
                'avc1',     # 기본 H.264 코덱
                'h264',     # 일반 H.264
                'h264_videotoolbox',  # macOS 하드웨어 가속 
                'H264',     # 대문자 버전
                'X264',     # x264 라이브러리
                'mp4v'      # 최후의 대안
            ]
            
            success = False
            for codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    self.curr_video_writer = cv2.VideoWriter(
                        filepath, fourcc, fps, (width, height))
                    
                    if self.curr_video_writer.isOpened():
                        logger.info(f"[{self.cctv_id}] 코덱 {codec_name} 초기화 성공")
                        success = True
                        break
                    else:
                        logger.warning(f"[{self.cctv_id}] 코덱 {codec_name} 초기화 실패")
                except Exception as e:
                    logger.warning(f"[{self.cctv_id}] 코덱 {codec_name} 사용 중 오류: {e}")
            
            if not success:
                # 모든 코덱이 실패한 경우 mp4v로 최종 시도
                logger.warning(f"[{self.cctv_id}] 모든 H.264 코덱 초기화 실패, mp4v 코덱으로 대체합니다.")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.curr_video_writer = cv2.VideoWriter(
                    filepath, fourcc, fps, (width, height))
            
            # 코덱 정보 로깅 (디버깅용)
            codec_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            logger.info(f"[{self.cctv_id}] 비디오 저장 시작 - 해상도: {width}x{height}, FPS: {fps}, 코덱: {codec_str}")
            
            self.curr_video_path = filepath
            self.curr_video_start_time = time.time()
            
            # 버퍼에 있는 이전 프레임들도 저장
            for buffered_frame in self.frame_buffer:
                self.curr_video_writer.write(buffered_frame)
                
            logger.info(f"[{self.cctv_id}] 🎥 녹화 시작: {filename}")
                
    async def stop_recording(self):
        """현재 비디오 녹화 종료 및 저장"""
        if self.curr_video_writer is not None:
            import os  # 함수 내에서 명시적으로 import
            self.curr_video_writer.release()
            logger.info(f"[{self.cctv_id}] 🛑 녹화 종료: {os.path.basename(self.curr_video_path)}")
            
            # 필요한 경우 ffmpeg을 사용하여 비디오 변환 (mp4v -> h264)
            local_clip_path = self.curr_video_path
            output_path = None
            
            # 녹화된 비디오의 코덱 확인
            try:
                cap = cv2.VideoCapture(local_clip_path)
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec_str = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
                cap.release()
                
                # mp4v 코덱으로 녹화된 경우, ffmpeg으로 h264로 변환
                if codec_str == 'mp4v':
                    logger.info(f"[{self.cctv_id}] mp4v 코덱 감지, ffmpeg으로 h264 변환 시도...")
                    output_path = local_clip_path.replace('.mp4', '_h264.mp4')
                    
                    # ffmpeg 명령어 실행
                    try:
                        import subprocess
                        cmd = [
                            'ffmpeg', '-i', local_clip_path, 
                            '-c:v', 'libx264', '-preset', 'fast', 
                            '-crf', '22', '-y', output_path
                        ]
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = process.communicate()
                        
                        if process.returncode == 0:
                            logger.info(f"[{self.cctv_id}] H.264 변환 성공: {output_path}")
                            # 변환 성공 시 원본 파일을 대체
                            os.replace(output_path, local_clip_path)
                            logger.info(f"[{self.cctv_id}] 원본 파일을 H.264 버전으로 교체")
                        else:
                            logger.warning(f"[{self.cctv_id}] H.264 변환 실패: {stderr.decode('utf-8', errors='ignore')}")
                            # 변환 실패 시 원본 사용
                            if os.path.exists(output_path):
                                os.remove(output_path)
                    except Exception as e:
                        logger.error(f"[{self.cctv_id}] ffmpeg 변환 중 오류: {e}")
                        if output_path and os.path.exists(output_path):
                            os.remove(output_path)
            except Exception as e:
                logger.error(f"[{self.cctv_id}] 코덱 확인 중 오류: {e}")
            
            # 나머지 S3 업로드 코드는 그대로 유지
            # 썸네일 생성
            thumbnail_path = None
            try:
                # 영상에서 썸네일 추출
                thumbnail_path = self._create_thumbnail(local_clip_path)
                logger.info(f"[{self.cctv_id}] 📸 썸네일 생성 완료: {thumbnail_path}")
            except Exception as e:
                logger.error(f"[{self.cctv_id}] 썸네일 생성 중 오류: {str(e)}")
            
            # S3 업로드 시도 (비동기)
            try:
                logger.info(f"[{self.cctv_id}] S3 업로드 시작: {local_clip_path}")
                
                # presigned URL 사용하여 업로드 및 URL 생성
                s3_key = f"clips/{os.path.basename(local_clip_path)}"
                url = await upload_to_s3(
                    local_clip_path,
                    s3_key,
                    expiry_seconds=604800,  # 7일 유효
                )
                
                # 썸네일 업로드
                thumbnail_url = None
                if thumbnail_path and os.path.exists(thumbnail_path):
                    thumbnail_s3_key = f"thumbnails/{os.path.basename(thumbnail_path)}"
                    
                    thumbnail_url = await upload_to_s3(
                        thumbnail_path,
                        thumbnail_s3_key,
                        expiry_seconds=604800,  # 7일 유효
                    )
                    
                    if thumbnail_url:
                        log_s3_event(True, thumbnail_path, thumbnail_s3_key, thumbnail_url)
                    else:
                        log_s3_event(False, thumbnail_path, thumbnail_s3_key)
                
                if url:
                    log_s3_event(True, local_clip_path, s3_key, url)
                    
                    # 외부 서버에 감지 정보 전송
                    detection_info = format_detection_for_api(
                        self.cctv_id,  # cctv_id (숫자 형식 그대로 전달)
                        url,  # videoUrl
                        self.last_behavior_type if self.last_behavior_type else "이상 행동 감지",  # anomalyType
                        self.last_confidence,  # confidence
                        None,  # timestamp
                        thumbnail_url,  # thumbnail_url
                        self.user_id  # user_id
                    )
                    await send_detection_info_to_server(detection_info)
                else:
                    log_s3_event(False, local_clip_path, s3_key)
            except Exception as e:
                logger.error(f"[{self.cctv_id}] S3 업로드 또는 API 전송 중 오류: {str(e)}")
            
            # 변수 초기화
            self.curr_video_writer = None
            self.curr_video_path = None
            self.curr_video_start_time = None

    def _create_thumbnail(self, video_path):
        """비디오에서 썸네일 이미지 생성"""
        # 썸네일 저장 디렉토리
        thumbnail_dir = os.path.join(self.recording_dir, "thumbnails")
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # 썸네일 파일 경로
        video_filename = os.path.basename(video_path)
        thumbnail_filename = video_filename.rsplit('.', 1)[0] + ".jpg"
        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"썸네일 생성을 위한 비디오 파일을 열 수 없습니다: {video_path}")
        
        # 비디오 정보 가져오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 비디오의 중간 프레임으로 이동 (더 의미있는 썸네일을 위해)
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        
        # 프레임 읽기
        success, frame = cap.read()
        if not success:
            # 중간 프레임을 읽을 수 없으면 첫 프레임 시도
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
            
        # 캡처 객체 해제
        cap.release()
        
        if not success:
            raise Exception(f"비디오에서 프레임을 추출할 수 없습니다: {video_path}")
        
        # 썸네일 크기 조정 (옵션)
        max_size = 480  # 최대 너비 또는 높이
        h, w = frame.shape[:2]
        if h > max_size or w > max_size:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            frame = cv2.resize(frame, (new_w, new_h))
        
        # 썸네일 저장
        cv2.imwrite(thumbnail_path, frame)
        
        return thumbnail_path

    def stop(self):
        """비디오 처리 종료"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            
        if self.curr_video_writer is not None:
            self.curr_video_writer.release()
            
        logger.info(f"[{self.cctv_id}] 스트림 처리 종료")

# --- API Routes ---

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 호출되는 이벤트 핸들러"""
    logger.warning("🚀 무인점포 절도 감지 서버 시작")
    
    # AWS 자격 증명 정보 출력
    print_aws_credentials_info()
    
    # S3 연결 테스트 (서버 시작시 한 번만 실행)
    try:
        result = await test_s3_connection()
        if result:
            logger.warning("✅ S3 연결 테스트 성공")
            
            # CORS 설정은 AWS 콘솔에서 이미 구성되어 있음
            await setup_bucket_cors()
            
            # S3 버킷의 공개 액세스 설정 확인
            try:
                public_access = await check_bucket_public_access()
                if public_access is False:
                    logger.warning("⚠️ S3 버킷에 공개 읽기 액세스가 설정되어 있지 않습니다.")
                    logger.warning("⚠️ 브라우저에서 미디어 파일을 직접 재생하려면 다음 버킷 정책을 추가하세요:")
                    logger.warning("""
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::cctv-recordings-yuhan-20250505/*"
        }
    ]
}
                    """)
                    logger.warning("⚠️ 또한 버킷의 '퍼블릭 액세스 차단' 설정도 비활성화해야 합니다.")
                elif public_access is True:
                    logger.warning("✅ S3 버킷에 공개 읽기 액세스가 올바르게 설정되어 있습니다.")
            except Exception as e:
                logger.error(f"S3 버킷 공개 액세스 확인 중 오류: {str(e)}")
        else:
            logger.warning("⚠️ S3 연결 테스트 실패 - 서버는 계속 실행됩니다")
    except Exception as e:
        logger.error(f"❌ S3 연결 테스트 중 오류: {str(e)} - 서버는 계속 실행됩니다")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 호출되는 이벤트 핸들러"""
    logger.warning("🛑 무인점포 절도 감지 서버 종료 중...")
    
    # 모든 비디오 프로세서 종료
    for processor in active_processors.values():
        processor.stop()

@app.get("/api/v1/status")
async def get_status():
    """서버 상태 확인 API"""
    running_streams = list(active_processors.keys())
    return {
        "status": "running",
        "message": "무인점포 절도 감지 시스템 실행 중",
        "active_streams": running_streams
    }

@app.get("/api/v1/active_streams")
async def get_active_streams():
    """현재 실행 중인 모든 스트림 목록 반환"""
    streams = []
    for cctv_id, processor in active_processors.items():
        streams.append({
            "cctv_id": cctv_id,
            "user_id": processor.user_id,
            "rtsp_url": processor.rtsp_url,
            "is_running": processor.is_running,
            "recording": processor.curr_video_writer is not None
        })
    
    return {"count": len(streams), "streams": streams}

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

@app.get("/api/v1/test/s3")
async def test_s3_connection_endpoint():
    """S3 연결 테스트 API"""
    try:
        # AWS 자격 증명 정보 출력
        creds_info = print_aws_credentials_info()
        
        # S3 연결 테스트
        s3_connected = await test_s3_connection()
        
        # 테스트 파일 생성 및 업로드 시도
        if s3_connected:
            # 테스트 파일 생성
            test_dir = "s3_test"
            os.makedirs(test_dir, exist_ok=True)
            test_file_path = f"{test_dir}/s3_test_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(test_file_path, "w") as f:
                f.write(f"S3 테스트 파일 - 생성시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 업로드 시도 - Content-Type 자동 설정됨
            s3_key = f"tests/s3_test_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            url = await upload_to_s3(test_file_path, s3_key)
            
            # 테스트 파일 삭제
            os.remove(test_file_path)
            
            if url:
                return {
                    "status": "success",
                    "message": "S3 연결 및 업로드 테스트 성공",
                    "credentials": creds_info,
                    "url": url
                }
            else:
                return {
                    "status": "partial_success",
                    "message": "S3 연결은 성공했으나 파일 업로드 실패",
                    "credentials": creds_info
                }
        else:
            return {
                "status": "failed",
                "message": "S3 연결 테스트 실패",
                "credentials": creds_info
            }
    except Exception as e:
        logger.error(f"S3 테스트 중 오류: {str(e)}")
        return {
            "status": "error",
            "message": f"S3 테스트 중 오류 발생: {str(e)}"
        }

@app.get("/api/v1/test/model")
async def test_model():
    """AI 모델 로딩 테스트 API"""
    try:
        # 테스트 이미지 생성 (검은색 배경의 빈 이미지)
        test_image = np.zeros((640, 480, 3), dtype=np.uint8)
        
        # 빈 keypoints_buffer 생성
        test_keypoints_buffer = collections.deque(maxlen=60)
        
        # 모델 테스트 실행 (keypoints_buffer 인자 추가)
        result = theft_detection_model(test_image, test_keypoints_buffer)
        
        # 결과 분석
        if result and isinstance(result, tuple) and len(result) >= 3:
            is_anomaly, confidence, behavior_type = result
            
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
        logger.error(f"AI 모델 테스트 중 오류: {str(e)}")
        import traceback
        return {
            "status": "error",
            "message": f"AI 모델 테스트 중 오류 발생: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.get("/api/v1/test/media")
async def test_media_upload():
    """미디어 파일(동영상/이미지) 업로드 테스트 API"""
    try:
        # 테스트 비디오와 이미지 파일 생성
        test_dir = "s3_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # 테스트 이미지 생성 (검은색 배경에 흰색 텍스트)
        test_image_path = f"{test_dir}/test_image_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # OpenCV로 텍스트 추가
        cv2.putText(img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(test_image_path, img)
        
        # 테스트 비디오 생성 (2초 길이)
        test_video_path = f"{test_dir}/test_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        fps = 30
        writer = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
        
        if not writer.isOpened():
            return {
                "status": "error", 
                "message": "비디오 파일을 생성할 수 없습니다."
            }
            
        # 2초 분량의 프레임 생성
        for i in range(60):  # 30fps * 2초 = 60프레임
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 프레임 번호와 시간 표시
            cv2.putText(frame, f"Frame: {i+1}/60", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Test Video", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
            
        writer.release()
        
        # S3에 업로드
        image_s3_key = f"tests/images/{os.path.basename(test_image_path)}"
        video_s3_key = f"tests/videos/{os.path.basename(test_video_path)}"
        
        image_url = await upload_to_s3(test_image_path, image_s3_key)
        video_url = await upload_to_s3(test_video_path, video_s3_key)
        
        # 로컬 테스트 파일 삭제
        os.remove(test_image_path)
        os.remove(test_video_path)
        
        return {
            "status": "success",
            "message": "미디어 파일 업로드 테스트 완료",
            "image": {
                "url": image_url,
                "content_type": "image/jpeg"
            },
            "video": {
                "url": video_url,
                "content_type": "video/mp4"
            },
            "html_preview": f"""
            <!DOCTYPE html>
            <html>
            <head><title>미디어 파일 테스트</title></head>
            <body>
                <h2>이미지 테스트</h2>
                <img src="{image_url}" alt="테스트 이미지" style="max-width: 100%">
                
                <h2>비디오 테스트</h2>
                <video width="640" height="480" controls>
                    <source src="{video_url}" type="video/mp4">
                    브라우저가 비디오 태그를 지원하지 않습니다.
                </video>
            </body>
            </html>
            """
        }
    except Exception as e:
        logger.error(f"미디어 테스트 중 오류: {str(e)}")
        import traceback
        return {
            "status": "error",
            "message": f"미디어 테스트 중 오류 발생: {str(e)}",
            "traceback": traceback.format_exc()
        }


# 하위 호환성을 위한 리디렉션 라우트
@app.get("/")
async def read_root():
    """API 루트 경로 (하위 호환성용)"""
    return await get_status()

@app.get("/active_streams")
async def legacy_get_active_streams():
    """하위 호환성을 위한 리디렉션"""
    return await get_active_streams()

@app.post("/start_stream")
async def legacy_start_stream(payload: dict = Body(...)):
    """하위 호환성을 위한 리디렉션"""
    # stream_id를 cctv_id로 변환
    if "stream_id" in payload and "cctv_id" not in payload:
        payload["cctv_id"] = payload["stream_id"]
    return await start_stream(payload)

# @app.post("/stop_stream/{stream_id}")
# async def legacy_stop_stream(stream_id: str):
#     """하위 호환성을 위한 리디렉션"""
#     return await stop_stream(stream_id)

@app.get("/test_s3")
async def legacy_test_s3():
    """하위 호환성을 위한 리디렉션"""
    return await test_s3_connection_endpoint()

@app.get("/test_model")
async def legacy_test_model():
    """하위 호환성을 위한 리디렉션"""
    return await test_model()

@app.get("/test_media")
async def legacy_test_media():
    """하위 호환성을 위한 리디렉션"""
    return await test_media_upload()

# # 서버 시작용 코드
# if __name__ == "__main__":
#     # 환경변수에서 포트 가져오기 (기본값: 8000)
#     port = int(os.environ.get("PORT", 8000))
    
#     print(f"서버 시작 중... 포트: {port}")
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

# cd venv && uvicorn main:app --reload --host 0.0.0.0 --port 8000


#    lsof -i :8000

#    ps aux | grep uvicorn



#    kill 56593