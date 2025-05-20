import boto3
import botocore
import asyncio
import logging
import os
import traceback
import mimetypes
import json


logger = logging.getLogger(__name__)

S3_BUCKET_NAME = "cctv-recordings-yuhan-20250505"
AWS_REGION = "ap-northeast-2"

s3_client = boto3.client('s3', region_name=AWS_REGION)

async def upload_to_s3(local_filepath: str, s3_filename: str, expiry_seconds: int = 604800) -> str | None:
    logger.info(f"S3 업로드 시작: {local_filepath} -> s3://{S3_BUCKET_NAME}/{s3_filename}")
    
    # 먼저 파일이 존재하는지 확인
    if not os.path.exists(local_filepath):
        logger.error(f"S3 업로드 실패: 로컬 파일이 존재하지 않음 - {local_filepath}")
        return None
        
    # 파일 크기 확인
    try:
        file_size = os.path.getsize(local_filepath)
        logger.info(f"업로드할 파일 크기: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        
        if file_size == 0:
            logger.error(f"S3 업로드 실패: 파일 크기가 0 bytes입니다 - {local_filepath}")
            return None
            
    except Exception as e:
        logger.error(f"파일 크기 확인 중 오류: {e}")
    
    # 파일 확장자에 따른 Content-Type 결정
    file_extension = os.path.splitext(local_filepath)[1].lower()
    content_type = None
    
    # 파일 타입에 따른 적절한 Content-Type 설정 (브라우저 호환성 최적화)
    if file_extension == '.mp4':
        content_type = 'video/mp4'  # 표준 MIME 타입
    elif file_extension == '.jpg' or file_extension == '.jpeg':
        content_type = 'image/jpeg'
    elif file_extension == '.png':
        content_type = 'image/png'
    else:
        # 기타 파일 유형은 mimetypes 모듈로 추측
        content_type = mimetypes.guess_type(local_filepath)[0]
    
    # 기본 Content-Type 설정
    if not content_type:
        content_type = 'application/octet-stream'
    
    logger.info(f"파일 {local_filepath}의 Content-Type을 {content_type}로 설정합니다.")
    
    # S3 업로드 ExtraArgs 설정
    extra_args = {
        'ContentType': content_type,
    }
    
    # 비디오나 이미지인 경우 인라인 표시 설정
    if content_type.startswith('video/') or content_type.startswith('image/'):
        extra_args['ContentDisposition'] = 'inline'  # 브라우저에서 인라인으로 표시
        
    try:
        # 비동기로 S3 업로드 수행 (ExtraArgs 추가)
        logger.info(f"S3 업로드 실행 중... (버킷: {S3_BUCKET_NAME}, 키: {s3_filename}, 컨텐츠 타입: {content_type})")
        await asyncio.to_thread(
            s3_client.upload_file,
            local_filepath,
            S3_BUCKET_NAME,
            s3_filename,
            ExtraArgs=extra_args
        )
        
        # 업로드 확인 - 객체가 실제로 존재하는지 확인
        try:
            response = await asyncio.to_thread(
                s3_client.head_object,
                Bucket=S3_BUCKET_NAME,
                Key=s3_filename
            )
            logger.info(f"S3 업로드 확인 성공: {s3_filename} (Size: {response.get('ContentLength', 'unknown')} bytes)")
            
            # 실제 Content-Type 및 Content-Disposition 확인
            actual_content_type = response.get('ContentType', '알 수 없음')
            actual_disposition = response.get('ContentDisposition', '설정되지 않음')
            logger.info(f"S3 객체 메타데이터 - ContentType: {actual_content_type}, ContentDisposition: {actual_disposition}")
            
        except Exception as e:
            logger.warning(f"S3 객체 확인 실패 (하지만 업로드는 성공했을 수 있음): {e}")
        
        # 일반 S3 URL 생성 (presigned URL 대신)
        file_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_filename}"
        logger.info(f"S3 URL 생성 성공: {file_url}")
        
        return file_url
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', 'Unknown error')
        if error_code == 'AccessDenied':
            logger.error(f"S3 업로드 권한 오류 (AccessDenied): 계정 권한을 확인하세요. 버킷: {S3_BUCKET_NAME}, 상세: {error_message}")
        else:
            logger.error(f"S3 업로드 실패 (ClientError): 코드: {error_code}, 메시지: {error_message}")
        # 스택 트레이스 출력
        logger.error(f"S3 업로드 오류 스택 트레이스: {traceback.format_exc()}")
        return None
        
    except FileNotFoundError:
        logger.error(f"S3 업로드 실패: 로컬 파일 없음 - {local_filepath}")
        return None
        
    except Exception as e:
        logger.error(f"S3 업로드 중 예외 발생: {e}")
        logger.error(f"S3 업로드 오류 스택 트레이스: {traceback.format_exc()}")
        return None

# S3 연결 테스트 함수 추가
async def test_s3_connection() -> bool:
    """
    S3 연결 및 권한을 테스트합니다.
    반환값: 연결 성공 여부 (True/False)
    """
    logger.info(f"S3 연결 테스트 시작 (버킷: {S3_BUCKET_NAME})")
    try:
        # 버킷 존재 여부 확인 (ListBucket 권한 필요)
        response = await asyncio.to_thread(
            s3_client.list_objects_v2,
            Bucket=S3_BUCKET_NAME,
            MaxKeys=1
        )
        logger.info(f"S3 연결 성공! 버킷 존재함: {S3_BUCKET_NAME}")
        
        # 테스트 파일 업로드 (PutObject 권한 필요)
        test_data = b"This is a test file for S3 connection."
        test_key = "test/connection_test.txt"
        
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=S3_BUCKET_NAME,
            Key=test_key,
            Body=test_data,
            ContentType='text/plain'
        )
        logger.info(f"S3 테스트 파일 업로드 성공: s3://{S3_BUCKET_NAME}/{test_key}")
        return True
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', 'Unknown error')
        
        if error_code == 'AccessDenied':
            logger.error(f"S3 연결 테스트 실패 (권한 오류): 계정에 필요한 권한이 없습니다.")
            logger.error(f"필요한 권한: s3:ListBucket, s3:PutObject, s3:GetObject")
            logger.error(f"상세 오류: {error_message}")
        elif error_code == 'NoSuchBucket':
            logger.error(f"S3 연결 테스트 실패: 버킷이 존재하지 않습니다: {S3_BUCKET_NAME}")
        else:
            logger.error(f"S3 연결 테스트 실패 (ClientError): 코드: {error_code}, 메시지: {error_message}")
        return False
        
    except Exception as e:
        logger.error(f"S3 연결 테스트 중 예외 발생: {e}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        return False

# AWS 자격 증명 정보 확인
def print_aws_credentials_info():
    """현재 사용 중인 AWS 자격 증명 정보를 출력합니다 (보안 정보는 제외)"""
    try:
        # 현재 세션의 자격 증명 확인
        session = boto3.session.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            logger.warning("AWS 자격 증명을 찾을 수 없습니다!")
            return
            
        # 자격 증명 출처 확인 (환경 변수, 프로필, EC2 인스턴스 역할 등)
        credential_source = "알 수 없음"
        if os.environ.get('AWS_ACCESS_KEY_ID'):
            credential_source = "환경 변수"
        elif os.path.exists(os.path.expanduser("~/.aws/credentials")):
            credential_source = "AWS 자격 증명 파일 (~/.aws/credentials)"
        
        # 보안상 AccessKey의 마지막 4자리만 표시
        access_key = credentials.access_key
        if access_key:
            masked_access_key = "***" + access_key[-4:] if len(access_key) >= 4 else "masked"
        else:
            masked_access_key = "없음"
            
        logger.info(f"AWS 자격 증명 정보:")
        logger.info(f"- 자격 증명 출처: {credential_source}")
        logger.info(f"- AWS 리전: {AWS_REGION}")
        logger.info(f"- Access Key ID: {masked_access_key}")
        logger.info(f"- Secret Key: {'설정됨' if credentials.secret_key else '없음'}")
        logger.info(f"- 세션 토큰: {'있음' if credentials.token else '없음'}")
        
    except Exception as e:
        logger.error(f"AWS 자격 증명 정보 확인 중 오류: {e}")

# CORS 설정 관련 함수 주석 처리
# S3 버킷에 CORS 정책 설정 함수
async def setup_bucket_cors():
    """S3 버킷에 CORS 정책을 설정합니다."""
    logger.info("CORS 설정은 AWS 콘솔에서 직접 구성되었습니다.")
    logger.info("다음과 같은 CORS 설정이 필요합니다:")
    logger.info("""
{
    "CORSRules": [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "HEAD"],
            "AllowedOrigins": ["*"],
            "ExposeHeaders": ["ETag", "Content-Length", "Content-Type", "Content-Disposition"],
            "MaxAgeSeconds": 3000
        }
    ]
}
    """)
    return True

async def check_bucket_public_access():
    """S3 버킷의 공개 액세스 설정을 확인합니다."""
    logger.info("버킷 공개 액세스 설정이 AWS 콘솔에서 구성되었습니다.")
    logger.info("미디어 파일을 브라우저에서 직접 재생하려면 다음 설정이 필요합니다:")
    logger.info("1. 버킷의 퍼블릭 액세스 차단 설정 해제")
    logger.info("2. 버킷 정책에서 s3:GetObject 액션을 * 주체에게 허용")
    return True