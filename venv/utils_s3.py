import boto3
import botocore
import asyncio
import logging
import os
import traceback


logger = logging.getLogger(__name__)

S3_BUCKET_NAME = "cctv-recordings-yuhan-20250505"
AWS_REGION = "ap-northeast-2"

s3_client = boto3.client('s3', region_name=AWS_REGION)

async def upload_to_s3(local_filepath: str, s3_filename: str) -> str | None:
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
        
    try:
        # 비동기로 S3 업로드 수행
        logger.info(f"S3 업로드 실행 중... (버킷: {S3_BUCKET_NAME}, 키: {s3_filename})")
        await asyncio.to_thread(
            s3_client.upload_file,
            local_filepath,
            S3_BUCKET_NAME,
            s3_filename
        )
        
        # 업로드 확인 - 객체가 실제로 존재하는지 확인
        try:
            response = await asyncio.to_thread(
                s3_client.head_object,
                Bucket=S3_BUCKET_NAME,
                Key=s3_filename
            )
            logger.info(f"S3 업로드 확인 성공: {s3_filename} (Size: {response.get('ContentLength', 'unknown')} bytes)")
        except Exception as e:
            logger.warning(f"S3 객체 확인 실패 (하지만 업로드는 성공했을 수 있음): {e}")
        
        file_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_filename}"
        logger.info(f"S3 업로드 성공: {file_url}")
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
            Body=test_data
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