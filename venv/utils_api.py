import httpx
import logging
import time
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# 스프링부트 서버 API 엔드포인트 설정
MAIN_SERVER_API_URL = "http://localhost:8080/api/anomaly/notify"  # 팀원의 코드와 일치하도록 변경

async def send_detection_info_to_server(detection_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    감지 정보(S3 URL 포함)를 스프링부트 서버 API로 전송합니다.

    Args:
        detection_data: API가 요구하는 형식의 딕셔너리 데이터
                        (예: {"cctvId": "cctv_01", "videoUrl": "...", 
                              "eventType": "THEFT", "eventTime": "2025-05-04T17:08:05"})
    
    Returns:
        Tuple[bool, Dict]: (성공 여부, 응답 데이터 또는 오류 정보)
    """
    logger.info(f"스프링부트 서버 API 호출 시작: {MAIN_SERVER_API_URL}")
    logger.info(f"전송 데이터: {detection_data}")
    
    try:
        async with httpx.AsyncClient() as client:
            # API 요청 보내기 (타임아웃 10초 설정)
            response = await client.post(
                MAIN_SERVER_API_URL, 
                json=detection_data, 
                timeout=10.0,
                headers={"Content-Type": "application/json"}
            )
            
            # 응답 상태 코드 확인 (2xx가 아니면 예외 발생)
            response.raise_for_status()
            
            # 성공 응답 처리
            logger.info(f"API 호출 성공: 상태 코드 {response.status_code}")
            
            # 응답이 JSON인지 확인
            response_data = {}
            try:
                response_data = response.json()
                logger.info(f"API 응답: {response_data}")
            except ValueError:
                logger.warning("API 응답이 JSON 형식이 아닙니다.")
                response_data = {"text": response.text}
                
            return True, response_data
            
    except httpx.HTTPStatusError as e:
        # HTTP 오류 응답 (4xx, 5xx)
        logger.error(f"API 호출 실패 (HTTP Status Error): 상태 코드 {e.response.status_code}")
        error_detail = ""
        try:
            error_detail = e.response.json()
        except ValueError:
            error_detail = e.response.text
        logger.error(f"오류 응답: {error_detail}")
        return False, {"error": f"HTTP Status Error: {e.response.status_code}", "detail": error_detail}
        
    except httpx.RequestError as e:
        # 요청 자체 실패 (네트워크 문제, 타임아웃 등)
        logger.error(f"API 호출 실패 (Request Error): {e}")
        return False, {"error": f"Request Error: {str(e)}"}
        
    except Exception as e:
        # 기타 예외 상황
        logger.error(f"API 호출 중 예외 발생: {e}", exc_info=True)
        return False, {"error": f"Unexpected Error: {str(e)}"}

def format_detection_for_api(
    cctv_id: str, 
    videoUrl: str, 
    anomalyType: str,
    confidence: float,
    timestamp: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    API 호출을 위해 감지 정보를 포맷팅합니다.
    
    Args:
        cctv_id: CCTV 식별자 
        videoUrl: S3에 업로드된 영상 URL
        anomalyType: 감지된 이벤트 유형 (예: "절도_행위", "이상_행동")
        confidence: 감지 신뢰도 점수
        timestamp: 이벤트 발생 시간 (ISO 8601 형식, 없으면 현재 시간)
        thumbnail_url: 썸네일 이미지 URL (선택 사항)
        user_id: 사용자 ID (필수, 기본값 설정 필요)
        
    Returns:
        Dict: API 요청 형식에 맞는 데이터 딕셔너리
    """
    # 이벤트 시간 ISO 8601 형식 (없으면 현재 시간 사용)
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # 사용자 ID가 없으면 기본값 설정 (필요에 따라 조정)
    if user_id is None:
        user_id = 1  # 기본 사용자 ID
        
    # 기본 데이터 구성
    api_data = {
        "videoUrl": videoUrl,
        "anomalyType": anomalyType,
        "timestamp": timestamp,
        "userId": user_id,
        "cctv_id": cctv_id  # cctv_id 필드 추가
    }
    
    # 썸네일 URL이 있으면 추가 (확장 기능)
    if thumbnail_url:
        api_data["thumbnailUrl"] = thumbnail_url
    
    # 신뢰도 정보 추가 (확장 기능)
    api_data["confidence"] = confidence
        
    return api_data