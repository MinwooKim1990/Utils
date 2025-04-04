import os
import ffmpeg
import argparse

def get_video_info(input_path):
    info = ffmpeg.probe(input_path)
    video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
    if video_stream is None:
        raise ValueError("비디오 스트림을 찾을 수 없습니다.")
    
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    fps_str = video_stream.get('avg_frame_rate', '0/0')
    try:
        num, den = fps_str.split('/')
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        fps = 0.0

    sample_rate = None
    if audio_stream and 'sample_rate' in audio_stream:
        sample_rate = int(audio_stream['sample_rate'])
    
    return width, height, fps, sample_rate

def process_video(input_path, output_path, debug):
    try:
        width, height, fps, sample_rate = get_video_info(input_path)
    except Exception as e:
        print(f"{os.path.basename(input_path)} 정보를 읽는 중 오류 발생: {e}")
        return

    print(f"처리중: {os.path.basename(input_path)} (해상도: {width}x{height}, fps: {fps:.2f}, 오디오 샘플레이트: {sample_rate})")
    
    # GPU NVENC 인코딩 옵션
    vcodec = 'hevc_nvenc'
    video_opts = {'qp': 20, 'preset': 'p5'}  # 상황에 따라 조정

    # 오디오 샘플레이트 조건: 192kHz보다 크면 재샘플링, 아니면 그대로 복사
    if sample_rate is not None and sample_rate > 192000:
        acodec = 'aac'
        audio_opts = {'ar': 192000, 'audio_bitrate': '96k'}
    else:
        acodec = 'copy'
        audio_opts = {}

    try:
        in_stream = ffmpeg.input(input_path)
        
        # 비디오 필터 체인 구성:
        # 1. 해상도가 1080 이상이면 스케일 적용
        # 2. 픽셀 포맷을 nv12로 변환
        # 3. fps가 30 이상이면 fps 필터를 적용하여 30fps로 변환
        video = in_stream.video
        if height > 1080:
            video = video.filter('scale', -2, 1080)
        video = video.filter('format', 'nv12')
        if fps >= 30:
            video = video.filter('fps', fps=30)
        
        # 출력 인자: -pix_fmt nv12는 출력 인자에 추가하지 않고 필터 체인에서 처리
        output_args = dict(vcodec=vcodec, **video_opts)
        
        # 오디오 스트림이 있으면 함께 출력, 없으면 비디오만 출력
        if sample_rate is not None:
            out = ffmpeg.output(video, in_stream.audio, output_path,
                                acodec=acodec, **audio_opts, **output_args)
        else:
            out = ffmpeg.output(video, output_path, **output_args)
        
        # 디버그 모드일 경우 실행할 명령어 출력
        if debug:
            cmd = out.compile()
            print("실행할 ffmpeg 명령어:")
            print(" ".join(cmd))
        
        out = out.overwrite_output()
        stdout, stderr = out.run(capture_stdout=True, capture_stderr=True)
        
        if debug:
            print("ffmpeg stdout:")
            print(stdout.decode('utf-8'))
            print("ffmpeg stderr:")
            print(stderr.decode('utf-8'))
        
        os.remove(input_path)
        print(f"변환 완료 및 삭제됨: {os.path.basename(input_path)}")
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode()
        print(f"{os.path.basename(input_path)} 처리 중 오류 발생: {error_msg}")
        if debug:
            print("디버그 모드: 전체 에러 정보:")
            print(error_msg)

def main():
    parser = argparse.ArgumentParser(
        description="동영상 변환: 1080p 이상이면 1080p로, 오디오 샘플레이트 192kHz 이상이면 192kHz로, fps가 30 이상이면 필터 체인에서 30fps로 조정 (GPU NVENC 사용)"
    )
    parser.add_argument('--input_folder', type=str, default='input_folder', help="입력 폴더 경로")
    parser.add_argument('--output_folder', type=str, default='output_folder', help="출력 폴더 경로")
    parser.add_argument('--debug', action='store_true', help="디버그 모드 활성화: ffmpeg 명령어와 로그 출력")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    debug = args.debug

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']

    for filename in os.listdir(input_folder):
        if os.path.splitext(filename)[1].lower() in video_extensions:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_video(input_path, output_path, debug)

if __name__ == "__main__":
    main()
