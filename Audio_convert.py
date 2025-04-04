# %%
import moviepy.editor as mp
import os

file_path = "React_page/public/music/First.m4a"
full_path = os.path.normpath(os.path.join(os.getcwd(), file_path))
output_path = os.path.splitext(full_path)[0] + '.mp3'

try:
    clip = mp.AudioFileClip(full_path)
    clip.write_audiofile(output_path, bitrate="320k")
    print(f"변환 완료: {full_path} -> {output_path}")
except Exception as e:
    print(f"변환 실패: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='음성 파일을 MP3로 변환하는 스크립트')
    parser.add_argument('input_file', type=str, help='입력 음성 파일 경로')
    parser.add_argument('output_file', type=str, help='출력 MP3 파일 경로')
    
    args = parser.parse_args()
    
    file_path = args.input_file
    output_path = args.output_file
    
    full_path = os.path.normpath(os.path.join(os.getcwd(), file_path))
    output_path = os.path.splitext(full_path)[0] + '.mp3'
    
    try:
        clip = mp.AudioFileClip(full_path)
        clip.write_audiofile(output_path, bitrate="320k")
        print(f"변환 완료: {full_path} -> {output_path}")
    except Exception as e:
        print(f"변환 실패: {e}")