# %%
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def derive_key(password):
    """
    비밀번호를 SHA256 해시를 통해 32바이트 키로 변환합니다.
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    return hashlib.sha256(password).digest()

def pad(data):
    """
    PKCS#7 패딩을 적용합니다.
    """
    block_size = AES.block_size  # 일반적으로 16바이트
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len

def unpad(data):
    """
    PKCS#7 패딩을 제거합니다.
    """
    pad_len = data[-1]
    return data[:-pad_len]

def encrypt_string(plain_text, password):
    """
    평문을 암호화할 때 랜덤 IV를 사용하여 AES-256 CBC 모드로 암호화한 후,
    IV와 암호문을 결합해 base64로 인코딩된 문자열을 반환합니다.
    """
    key = derive_key(password)
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    if isinstance(plain_text, str):
        plain_text = plain_text.encode('utf-8')
    
    padded_text = pad(plain_text)
    ciphertext = cipher.encrypt(padded_text)
    
    # IV와 암호문을 결합합니다.
    encrypted_data = iv + ciphertext
    return base64.b64encode(encrypted_data).decode('utf-8')

def decrypt_string(encrypted_text, password):
    """
    암호화된 문자열을 복호화합니다.
    암호문 앞부분에 저장된 IV를 추출하여 사용합니다.
    """
    key = derive_key(password)
    encrypted_data = base64.b64decode(encrypted_text)
    
    # 처음 16바이트는 IV, 나머지는 암호문
    iv = encrypted_data[:AES.block_size]
    ciphertext = encrypted_data[AES.block_size:]
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext)
    decrypted = unpad(decrypted_padded)
    
    return decrypted.decode('utf-8')

def derive_key(password):
    """
    비밀번호를 SHA256 해시를 통해 32바이트 키로 변환합니다.
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    return hashlib.sha256(password).digest()

def derive_iv(password):
    """
    비밀번호를 MD5 해시를 통해 16바이트 IV로 변환합니다.
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    return hashlib.md5(password).digest()

def pad(data):
    """
    PKCS#7 방식으로 패딩을 적용합니다.
    """
    block_size = AES.block_size  # 일반적으로 16바이트
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len

def encrypt_string_deterministic(plain_text, password):
    """
    평문과 비밀번호를 받아, 결정적으로 AES-256 CBC 모드로 암호화한 후
    base64로 인코딩된 암호문을 반환합니다.
    
    - 키는 SHA256(password)로, IV는 MD5(password)로 생성합니다.
    - 이 방식은 같은 평문과 비밀번호에 대해 항상 같은 암호문을 생성합니다.
    """
    key = derive_key(password)
    iv = derive_iv(password)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    if isinstance(plain_text, str):
        plain_text = plain_text.encode('utf-8')
    
    padded_text = pad(plain_text)
    ciphertext = cipher.encrypt(padded_text)
    
    return base64.b64encode(ciphertext).decode('utf-8')


def encode_string(plaintext, password):
    """
    평문과 비밀번호를 입력받아 XOR 방식으로 인코딩한 후 16진수 문자열을 반환합니다.
    """
    plaintext_bytes = plaintext.encode('utf-8')
    password_bytes = password.encode('utf-8')
    result = bytearray()
    for i, b in enumerate(plaintext_bytes):
        result.append(b ^ password_bytes[i % len(password_bytes)])
    return result.hex()

def decode_string(encoded_hex, password):
    """
    16진수 문자열로 인코딩된 데이터를 비밀번호를 사용해 복호화하여 평문을 반환합니다.
    """
    encoded_bytes = bytearray.fromhex(encoded_hex)
    password_bytes = password.encode('utf-8')
    result = bytearray()
    for i, b in enumerate(encoded_bytes):
        result.append(b ^ password_bytes[i % len(password_bytes)])
    return result.decode('utf-8')

# 사용 예시
if __name__ == '__main__':
    API_KEY = "Local_Access::MinwooKim1990##"
    password = "2024"
    encoded = encode_string(API_KEY, password)
    print("인코딩 결과:", encoded)
    decoded = decode_string(encoded, password)
    print("디코딩 결과:", decoded)

# %%
