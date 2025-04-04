# %%
import os
import json
import hashlib
import subprocess
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import statistics
import random

def get_file_hash(file_path, chunk_size=16*1024*1024):  # 16MB 청크 사용
    """파일의 해시 값을 효율적으로 계산"""
    hasher = hashlib.md5()
    
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(chunk_size)
            while chunk:
                hasher.update(chunk)
                chunk = f.read(chunk_size)
        
        return hasher.hexdigest()
    except Exception as e:
        print(f"해시 계산 오류 {file_path}: {e}")
        return None

def get_quick_file_hash(file_path, sample_size=1024*1024):  # 1MB 샘플
    """빠른 해시 계산을 위해 파일의 시작, 중간, 끝 부분만 샘플링"""
    hasher = hashlib.md5()
    
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            # 파일 시작 부분 읽기
            start_chunk = f.read(min(sample_size, file_size))
            hasher.update(start_chunk)
            
            # 파일이 충분히 크면 중간 부분도 읽기
            if file_size > sample_size * 2:
                f.seek(file_size // 2 - sample_size // 2)
                mid_chunk = f.read(min(sample_size, file_size // 2))
                hasher.update(mid_chunk)
            
            # 파일이 충분히 크면 끝 부분도 읽기
            if file_size > sample_size:
                f.seek(max(0, file_size - sample_size))
                end_chunk = f.read(min(sample_size, file_size))
                hasher.update(end_chunk)
        
        return hasher.hexdigest()
    except Exception as e:
        print(f"빠른 해시 계산 오류 {file_path}: {e}")
        return None

def is_video_playable(file_path):
    """ffmpeg를 사용하여 비디오 파일이 재생 가능한지 확인"""
    try:
        # ffprobe 사용하여 비디오 스트림 확인 (타임아웃 설정)
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10  # 10초 타임아웃
        )
        return 'video' in result.stdout.lower()
    except Exception as e:
        print(f"비디오 확인 오류 {file_path}: {e}")
        return False

def is_media_or_archive(file_path):
    """파일이 비디오, 이미지 또는 압축 파일인지 확인"""
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    archive_exts = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'}
    
    ext = os.path.splitext(file_path.lower())[1]
    return ext in video_exts or ext in image_exts or ext in archive_exts

def is_video_file(file_path):
    """파일이 비디오인지 확인"""
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
    ext = os.path.splitext(file_path.lower())[1]
    return ext in video_exts

def scan_directory(root_dir):
    """디렉토리 스캔 - 결과 캐싱"""
    file_dict = {}
    file_sizes = []
    root_path = Path(root_dir)
    
    # 안전한 캐시 파일명 생성
    safe_name = root_dir.replace(':', '_')
    safe_name = safe_name.replace('/', '_')
    safe_name = safe_name.replace('\\', '_')
    cache_file = f"{safe_name}_scan_cache.json"
    
    # 캐시 파일이 있으면 로드
    if os.path.exists(cache_file) and os.path.getmtime(cache_file) > time.time() - 3600:  # 1시간 이내
        try:
            print(f"캐시 파일 '{cache_file}'에서 디렉토리 정보 로드 중...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data['files'], cached_data['sizes']
        except Exception as e:
            print(f"캐시 로드 실패: {e}")
    
    print(f"디렉토리 '{root_dir}' 스캔 중...")
    try:
        for path in tqdm(list(root_path.glob('**/*')), desc=f"{root_dir} 스캔 중"):
            if path.is_file() and is_media_or_archive(str(path)):
                rel_path = str(path.relative_to(root_path))
                file_dict[rel_path] = str(path)
                try:
                    size = path.stat().st_size
                    file_sizes.append(size)
                except Exception:
                    pass
    except Exception as e:
        print(f"디렉토리 스캔 오류: {e}")
    
    # 캐시 저장
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'files': file_dict,
                'sizes': file_sizes
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"캐시 저장 실패: {e}")
    
    return file_dict, file_sizes

def save_progress(checkpoint_file, completed, mismatched, missing=None):
    """진행 상황 저장"""
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'completed': completed,
                'mismatched': mismatched,
                'missing': missing or []
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"체크포인트 저장 실패: {e}")

def load_progress(checkpoint_file):
    """진행 상황 로드"""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('completed', []), data.get('mismatched', []), data.get('missing', [])
    except Exception as e:
        print(f"체크포인트 로드 실패: {e}")
    
    return [], [], []

def process_file_pair(item, quick_mode=True):
    """파일 쌍 처리 (빠른 모드 지원)"""
    rel_path, (source_path, target_path) = item
    
    try:
        # 파일 크기 비교 (빠른 필터링)
        try:
            source_size = os.path.getsize(source_path)
            target_size = os.path.getsize(target_path)
            
            if source_size != target_size:
                return rel_path, {
                    'relative_path': rel_path,
                    'source_path': source_path,
                    'target_path': target_path,
                    'source_size': source_size,
                    'target_size': target_size,
                    'mismatch_type': 'size'
                }
        except Exception as e:
            print(f"크기 확인 오류 {rel_path}: {e}")
        
        # 해시 계산 (빠른 모드 또는 일반 모드)
        hash_func = get_quick_file_hash if quick_mode else get_file_hash
        
        source_hash = hash_func(source_path)
        if source_hash is None:
            return rel_path, {
                'relative_path': rel_path,
                'source_path': source_path,
                'target_path': target_path,
                'error': '소스 파일 해시 계산 실패',
                'mismatch_type': 'error'
            }
        
        target_hash = hash_func(target_path)
        if target_hash is None:
            return rel_path, {
                'relative_path': rel_path,
                'source_path': source_path,
                'target_path': target_path,
                'error': '대상 파일 해시 계산 실패',
                'mismatch_type': 'error'
            }
        
        if source_hash != target_hash:
            mismatch_info = {
                'relative_path': rel_path,
                'source_path': source_path,
                'target_path': target_path,
                'source_hash': source_hash,
                'target_hash': target_hash,
                'mismatch_type': 'hash'
            }
            
            # 빠른 모드에서 불일치 감지하면 정밀 계산으로 다시 확인
            if quick_mode:
                full_source_hash = get_file_hash(source_path)
                full_target_hash = get_file_hash(target_path)
                
                if full_source_hash == full_target_hash:
                    # 정밀 계산에서는 일치하므로 거짓 양성
                    return rel_path, None
                
                mismatch_info['source_hash_full'] = full_source_hash
                mismatch_info['target_hash_full'] = full_target_hash
                
                # 비디오 파일이면 재생 가능 여부도 확인
                if is_video_file(rel_path):
                    mismatch_info['target_playable'] = is_video_playable(target_path)
            
            return rel_path, mismatch_info
        
        # 일치하는 경우
        return rel_path, None
    
    except Exception as e:
        return rel_path, {
            'relative_path': rel_path,
            'source_path': source_path,
            'target_path': target_path,
            'error': str(e),
            'mismatch_type': 'error'
        }

def compare_directories(source_dir, target_dir, output_json, checkpoint_file, 
                       num_workers=8, resume=False, quick_mode=True, 
                       sample_percent=None, max_files=None):
    """디렉토리 비교 메인 함수"""
    # 디렉토리 스캔 (캐싱 적용)
    source_files, source_sizes = scan_directory(source_dir)
    print(f"소스 디렉토리에서 {len(source_files)}개 파일 발견")
    
    target_files, _ = scan_directory(target_dir)
    print(f"대상 디렉토리에서 {len(target_files)}개 파일 발견")
    
    # 공통 파일 찾기
    common_files = {
        rel_path: (source_files[rel_path], target_files[rel_path])
        for rel_path in source_files
        if rel_path in target_files
    }
    
    print(f"비교할 공통 파일 {len(common_files)}개 발견")
    
    # 소스에만 있는 파일 (대상에 누락)
    missing_in_target = [
        rel_path for rel_path in source_files
        if rel_path not in target_files
    ]
    
    print(f"대상 디렉토리에 누락된 파일 {len(missing_in_target)}개 발견")
    
    # 이어하기
    completed_files = []
    mismatched_files = []
    if resume and os.path.exists(checkpoint_file):
        completed_files, mismatched_files, checkpoint_missing = load_progress(checkpoint_file)
        print(f"체크포인트에서 {len(completed_files)}개 파일 건너뛰기")
        
        # 이미 완료된 파일 제외
        for rel_path in completed_files:
            if rel_path in common_files:
                del common_files[rel_path]
    
    # 샘플링 모드
    if sample_percent is not None and 0 < sample_percent < 100:
        num_samples = max(1, int(len(common_files) * sample_percent / 100))
        sample_keys = random.sample(list(common_files.keys()), num_samples)
        common_files = {k: common_files[k] for k in sample_keys}
        print(f"전체 파일의 {sample_percent}%인 {len(common_files)}개 파일만 샘플링하여 처리")
    
    # 최대 파일 수 제한
    if max_files is not None and max_files > 0 and len(common_files) > max_files:
        sample_keys = random.sample(list(common_files.keys()), max_files)
        common_files = {k: common_files[k] for k in sample_keys}
        print(f"최대 {max_files}개 파일만 처리")
    
    # 처리 시간 예상
    if source_sizes:
        avg_file_size = statistics.mean(source_sizes) if source_sizes else 0
        est_time_per_file = 0.5 if quick_mode else 2.0  # 초 단위 추정
        total_time = len(common_files) * est_time_per_file / num_workers / 60  # 분 단위
        print(f"예상 비교 시간: {total_time:.2f}분 ({total_time/60:.2f}시간)")
        print(f"{'빠른' if quick_mode else '전체'} 해시 모드 사용 중")
    
    # 파일 목록 처리 준비
    file_items = list(common_files.items())
    total_files = len(file_items)
    
    # 병렬 처리 (배치 단위)
    batch_size = 10  # 한 번에 처리할 파일 수
    batch_count = (total_files + batch_size - 1) // batch_size
    
    print(f"총 {batch_count}개 배치 ({batch_size}개 파일/배치) 처리 중...")
    
    with tqdm(total=total_files, desc="파일 비교 중") as pbar:
        for i in range(0, batch_count):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            current_batch = file_items[start_idx:end_idx]
            
            # 배치 병렬 처리
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_item = {executor.submit(process_file_pair, item, quick_mode): item for item in current_batch}
                
                for future in concurrent.futures.as_completed(future_to_item):
                    try:
                        rel_path, mismatch_info = future.result()
                        
                        # 진행 상황 업데이트
                        completed_files.append(rel_path)
                        pbar.update(1)
                        
                        # 불일치 파일 기록
                        if mismatch_info:
                            mismatched_files.append(mismatch_info)
                    
                    except Exception as e:
                        print(f"작업 처리 오류: {e}")
            
            # 각 배치 후 체크포인트 저장
            if i % 10 == 0 or i == batch_count - 1:  # 10개 배치마다 또는 마지막 배치
                save_progress(checkpoint_file, completed_files, mismatched_files, missing_in_target)
                print(f"체크포인트 저장 완료 ({len(completed_files)}/{total_files})")
    
    # 결과 저장
    results = {
        'total_files_checked': len(completed_files),
        'missing_in_target': missing_in_target,
        'mismatched_files': mismatched_files
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 요약 출력
    print(f"\n결과 요약:")
    print(f"- 확인된 총 파일 수: {len(completed_files)}")
    print(f"- 누락된 파일 수: {len(missing_in_target)}")
    print(f"- 불일치 파일 수: {len(mismatched_files)}")
    
    # 불일치 유형 분석
    if mismatched_files:
        size_mismatches = sum(1 for info in mismatched_files if info.get('mismatch_type') == 'size')
        hash_mismatches = sum(1 for info in mismatched_files if info.get('mismatch_type') == 'hash')
        errors = sum(1 for info in mismatched_files if info.get('mismatch_type') == 'error')
        
        print(f"- 크기 불일치: {size_mismatches}")
        print(f"- 내용 불일치: {hash_mismatches}")
        print(f"- 오류 발생: {errors}")
        
        # 재생 불가능 비디오 확인
        unplayable_videos = sum(1 for info in mismatched_files 
                              if 'target_playable' in info and not info['target_playable'])
        if unplayable_videos > 0:
            print(f"- 재생 불가능 비디오: {unplayable_videos}")
    
    # 손상률 계산
    if completed_files:
        corruption_rate = len(mismatched_files) / len(completed_files) * 100
        print(f"손상률: {corruption_rate:.2f}%")
        
        if corruption_rate >= 1.0:
            print("\n경고: 손상률이 1% 이상입니다!")
            print("권장사항: 저장 매체 상태를 확인하고 파일을 다시 복사하는 것을 고려하세요.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="고속 파일 무결성 검사 도구")
    parser.add_argument("--source", required=True, help="소스 디렉토리")
    parser.add_argument("--target", required=True, help="대상 디렉토리")
    parser.add_argument("--output", default="file_comparison_results.json", help="결과 파일")
    parser.add_argument("--checkpoint", default="comparison_checkpoint.json", help="체크포인트 파일")
    parser.add_argument("--workers", type=int, default=8, help="작업자 스레드 수")
    parser.add_argument("--resume", action="store_true", help="이어하기")
    parser.add_argument("--full", action="store_true", help="전체 해시 모드 사용 (느림)")
    parser.add_argument("--sample", type=float, help="처리할 파일의 백분율 (예: 10)")
    parser.add_argument("--max-files", type=int, help="처리할 최대 파일 수")
    
    args = parser.parse_args()
    
    print("\n고속 파일 무결성 검사 도구")
    print("=" * 50)
    print(f"소스 디렉토리: {args.source}")
    print(f"대상 디렉토리: {args.target}")
    print(f"결과 파일: {args.output}")
    print(f"체크포인트 파일: {args.checkpoint}")
    print(f"{args.workers}개 작업자 스레드 사용")
    
    if args.resume:
        print("이전 체크포인트에서 이어하기")
    
    quick_mode = not args.full
    if not quick_mode:
        print("전체 해시 모드 사용 (느림)")
    
    if args.sample:
        print(f"샘플링 모드: {args.sample}% 파일만 처리")
    
    if args.max_files:
        print(f"최대 {args.max_files}개 파일만 처리")
    
    start_time = time.time()
    
    compare_directories(
        args.source, 
        args.target, 
        args.output, 
        args.checkpoint,
        args.workers,
        args.resume,
        quick_mode,
        args.sample,
        args.max_files
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")

if __name__ == "__main__":
    main()