# %%
from PIL import Image, ImageEnhance, ImageFilter
import os

# Pillow 라이브러리가 설치되어 있지 않다면 설치해야 합니다:
# pip install Pillow

def resize_image(input_path, output_path, resize_dims=None, dpi=None, brightness_factor=1.0, sharpen=False, unsharp_mask_params=None):
  """
  이미지 리사이즈, DPI, 밝기, 선명도(기본 또는 언샵마스크)를 선택적으로 조절하고 저장합니다.

  Args:
    input_path (str): 원본 이미지 파일 경로
    output_path (str): 저장할 이미지 파일 경로
    resize_dims (tuple, optional): (width, height) 튜플. None이면 리사이즈 안 함. 기본값 None.
    dpi (int, optional): 설정할 해상도 (DPI). None이면 설정 안 함. 기본값 None.
    brightness_factor (float, optional): 밝기 조절 계수. 1.0은 원본 밝기. 기본값 1.0.
    sharpen (bool, optional): True면 기본 선명화(SHARPEN) 적용. 기본값 False.
                               unsharp_mask_params가 설정되면 무시됨.
    unsharp_mask_params (dict, optional): UnsharpMask 파라미터 {'radius': r, 'percent': p, 'threshold': t}.
                                         None이면 적용 안 함. 기본값 None.
  """
  try:
    with Image.open(input_path) as img:
      processed_img = img.copy() # 원본을 유지하기 위해 복사본 사용

      # 1. 리사이즈
      if resize_dims:
        width, height = resize_dims
        try:
          resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
          resample_filter = Image.LANCZOS
        processed_img = processed_img.resize((width, height), resample=resample_filter)
        print(f"이미지를 {width}x{height} 크기로 리사이즈했습니다.")

      # 2. 밝기 조절
      if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(processed_img)
        processed_img = enhancer.enhance(brightness_factor)
        print(f"이미지 밝기를 {brightness_factor}배 조절했습니다.")

      # 3. 선명화 (언샵 마스크 우선 적용)
      if unsharp_mask_params:
        radius = unsharp_mask_params.get('radius', 2)
        percent = unsharp_mask_params.get('percent', 150)
        threshold = unsharp_mask_params.get('threshold', 3)
        processed_img = processed_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        print(f"UnsharpMask 필터 적용 (radius={radius}, percent={percent}, threshold={threshold})")
      elif sharpen:
        processed_img = processed_img.filter(ImageFilter.SHARPEN)
        print("기본 선명화(SHARPEN) 필터를 적용했습니다.")

      # 4. 저장 포맷 처리 (JPEG 변환)
      save_options = {}
      file_extension = os.path.splitext(output_path)[1].lower()

      if file_extension in ['.jpg', '.jpeg']:
        if processed_img.mode == 'RGBA':
          print(f"경고: RGBA 이미지를 JPEG로 저장하기 위해 RGB로 변환합니다. (투명도 제거됨) - {output_path}")
          processed_img = processed_img.convert('RGB')
        # JPEG 품질 설정 등 추가 옵션 가능 (예: save_options['quality'] = 95)

      # 5. DPI 설정 (저장 옵션에 추가)
      if dpi:
        save_options['dpi'] = (dpi, dpi)
        print(f"이미지 DPI를 {dpi}로 설정합니다.")

      # 이미지 저장
      processed_img.save(output_path, **save_options)
      print(f"이미지가 성공적으로 처리되어 '{output_path}'에 저장되었습니다.")

  except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다 - {input_path}")
  except Exception as e:
    print(f"오류 발생: {e}")

# --- 예제 사용법 ---
if __name__ == "__main__":
  input_image = "그림16.jpg"
  # 예제 4: 모든 기능 사용 (리사이즈, DPI, 밝기, 언샵마스크)
  output_4 = "16.jpg"
  print(f"\n--- 예제 4: {output_4} ---")
  resize_image(input_path=input_image,
               output_path=output_4,
               resize_dims=(413, 531),
               dpi=300,
               #brightness_factor=1,
               #sharpen = True, 
               #unsharp_mask_params={'radius': 1.5, 'percent': 120, 'threshold': 3}
               )
# %%
