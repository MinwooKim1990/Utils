import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor


def get_subfolder_names(folder_path):
    """지정된 폴더 안에 있는 하위 폴더들의 이름을 리스트로 반환"""
    try:
        # 폴더 내의 모든 항목을 가져옴
        items = os.listdir(folder_path)
        
        # 하위 폴더만 필터링
        subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        
        return subfolders
    except Exception as e:
        print(f"Error while reading folder: {e}")
        return []
    
def convert_mkv_to_mp4(input_file, output_file):
    """MKV 파일을 MP4로 변환 (코덱 복사 사용)"""
    command = ['ffmpeg', '-i', input_file, '-codec', 'copy', output_file]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted: {input_file} -> {output_file}")
        return True  # 변환 성공
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {input_file}: {e.stderr.decode()}")
        return False  # 변환 실패

def copy_file(src, dst):
    """파일을 복사하는 함수"""
    try:
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    except Exception as e:
        print(f"Failed to copy {src}: {e}")

def delete_file(file_path):
    """파일 삭제"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")

def process_file(mkv_file_path, conversion_folder):
    """개별 파일 처리 함수"""
    mp4_file_name = os.path.splitext(os.path.basename(mkv_file_path))[0] + '.mp4'
    mp4_file_path = os.path.join(conversion_folder, mp4_file_name)
    
    # MKV 파일을 MP4로 변환
    success = convert_mkv_to_mp4(mkv_file_path, mp4_file_path)
    
    if success:
        # 변환 성공 시, MP4 파일만 conversion 폴더로 이동
        print(f"MP4 file is sufficient: {mp4_file_path}")
    else:
        # 변환 실패 시, MKV 파일을 conversion 폴더로 복사
        mkv_copy_path = os.path.join(conversion_folder, os.path.basename(mkv_file_path))
        copy_file(mkv_file_path, mkv_copy_path)
        
        # 실패하면서 생성된 MP4 파일 삭제
        if os.path.exists(mp4_file_path):
            delete_file(mp4_file_path)

def process_folder(folder_path, max_workers=4):
    """폴더 내 모든 MKV 파일을 병렬로 처리"""
    # conversion 폴더 생성
    conversion_folder = os.path.join(os.path.dirname(folder_path), "test_conversion")
    os.makedirs(conversion_folder, exist_ok=True)
    print(f"Created conversion folder: {conversion_folder}")
    
    # MKV 파일 목록 가져오기
    mkv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mkv')]
    
    if not mkv_files:
        print("No MKV files found in the folder.")
        return
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mkv_file in mkv_files:
            executor.submit(process_file, mkv_file, conversion_folder)

if __name__ == "__main__":
    folder_path = "G:/new/1/haijiao"  # 폴더 경로 지정
    subfolder_names = get_subfolder_names(folder_path)

    print("Subfolders in the folder:")
    for subs in subfolder_names:
        folder_path = f"G:/new/1/haijiao/{subs}"
        max_workers = 4
        process_folder(folder_path, max_workers)
        print(f"All conversions and copies in {subs} completed.")
        is_con = input("Continue?: yes:1, no:2")
        if is_con == 1:
            continue
        elif is_con == 2:
            break
