# %%
def images_to_pdf(image_paths, output_pdf):
    """
    이미지 파일들을 하나의 PDF로 만드는 함수입니다.
    
    매개변수:
        image_paths (list of str): 이미지 파일 경로들의 리스트.
        output_pdf (str): 생성될 PDF 파일의 경로.
    
    사용 예시:
        images = ['image1.jpg', 'image2.png', 'image3.bmp']
        images_to_pdf(images, 'combined.pdf')
    """
    from PIL import Image

    if not image_paths:
        raise ValueError("이미지 파일 경로 리스트가 비어 있습니다.")
    
    # 모든 이미지를 RGB 모드로 변환 후 리스트에 저장
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    
    # 첫 번째 이미지를 기준으로 나머지 이미지를 추가하여 PDF 생성
    first_image = images[0]
    additional_images = images[1:]
    first_image.save(output_pdf, "PDF", resolution=100.0, save_all=True, append_images=additional_images)

images_to_pdf(['건보.jpg'],'아빠_건강보험료납부확인서.pdf')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='이미지 파일들을 하나의 PDF로 만드는 스크립트')
    parser.add_argument('image_paths', nargs='+', type=str, help='입력 이미지 파일 경로 리스트')
    parser.add_argument('output_pdf', type=str, help='출력 PDF 파일 경로')
    
    args = parser.parse_args()
    
    images_to_pdf(args.image_paths, args.output_pdf)