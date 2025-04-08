# %%
# --- 필요한 라이브러리 설치 ---
# pip install wsgidav cheroot

import os
from wsgidav.wsgidav_app import WsgiDAVApp
from wsgidav.fs_dav_provider import FilesystemProvider
from cheroot import wsgi

# --- 설정 ---

# 1. 공유할 경로 (D 드라이브 전체)
share_path = "D:/"

# 2. 사용자 인증 정보 (아이폰 앱의 '사용자', '암호' 항목에 해당)
#    보안을 위해 복잡한 사용자 이름과 암호를 사용하세요!
webdav_user = "kimminwoo190"  # 원하는 사용자 이름으로 변경하세요
webdav_pass = "1324k56k7" # 강력한 비밀번호로 변경하세요

# 3. 서버 주소 및 포트 (아이폰 앱의 '호스트', '포트' 항목에 해당)
listen_ip = "0.0.0.0"  # PC의 모든 IP 주소에서 접속 허용
listen_port = 443     # HTTP 포트 (기본 80 외 다른 포트 사용)

# 4. 읽기 전용 설정 (True로 설정 시 아이폰에서 파일 수정/삭제 불가)
read_only_mode = True # True: 읽기 전용, False: 읽기/쓰기 가능 (주의!)

# --- WsgiDAV 설정 ---
config = {
    "provider_mapping": {
        # 루트 경로('/')를 D 드라이브에 매핑
        "/": FilesystemProvider(share_path, readonly=read_only_mode),
    },
    "http_authenticator": {
        "domain_controller": None,  # simple_dc 사용
        "accept_basic": True,       # 기본 인증 허용
        "accept_digest": True,      # 다이제스트 인증 허용 (기본보다 약간 더 안전)
        "default_to_digest": True,  # 기본으로 다이제스트 사용
    },
    # 사용자 이름/암호 인증 설정
    "simple_dc": {
        "user_mapping": {
            "*": { # '*'는 모든 영역(realm)에 대해 적용
                webdav_user: {"password": webdav_pass}
            }
        }
    },
    "verbose": 1,  # 로그 레벨 (1: 기본 정보 표시)
    # --- 로깅 설정 섹션 ---
    "logging": {
         "enable_loggers": [], # 'logging' 하위 키로 이동
    },
    # --- 속성 저장소 (property_manager 대신 사용) ---
    "property_storage": True, # 이름 변경
    # --- 잠금 저장소 (lock_manager 대신 사용) ---
    "lock_storage": True,     # 이름 변경
}

# --- 서버 실행 ---
app = WsgiDAVApp(config)

# Cheroot WSGI 서버 설정 (HTTP)
server_args = {
    "bind_addr": (listen_ip, listen_port),
    "wsgi_app": app,
}

server = wsgi.Server(**server_args)

# 서버 시작 정보 출력
pc_ip_address = "PC의 IP 주소" # 실제 IP 주소로 확인 필요 (ipconfig 명령어 등)
print("=" * 50)
print(f"간단한 WebDAV 서버 (HTTP) 시작 중...")
print(f"!!! 경고: 이 서버는 HTTP를 사용하므로 통신이 암호화되지 않습니다.")
print(f"    신뢰할 수 있는 내부 네트워크에서만 사용하세요.")
print("-" * 50)
print(f"접속 주소: http://{pc_ip_address}:{listen_port}/")
print(f"공유 폴더: {share_path} ({'읽기 전용' if read_only_mode else '읽기/쓰기'})")
print(f"설정된 사용자 이름: {webdav_user}")
print("-" * 50)
print("아이폰 앱 설정:")
print(f"  - 제목: (자유롭게 입력)")
print(f"  - 호스트: {pc_ip_address}")
print(f"  - 사용자: {webdav_user}")
print(f"  - 암호: (설정한 비밀번호 입력)")
print(f"  - 암호 잠금: (앱 자체 기능, 필요시 활성화)")
print(f"  - 포트: {listen_port}")
print(f"  - 경로: / (또는 비워둠)")
print(f"  - HTTPS: 비활성화 (꺼짐)") # 중요! HTTPS 끄기
print("-" * 50)
print(f"!!! PC의 방화벽에서 {listen_port} 포트(TCP)를 열어야 할 수 있습니다.")
print("Ctrl+C 를 눌러 서버를 중지할 수 있습니다.")
print("=" * 50)

try:
    server.start()
except KeyboardInterrupt:
    print("\n서버 중지 요청됨...")
    server.stop()
    print("WebDAV 서버가 중지되었습니다.")
except Exception as e:
    print(f"\n오류 발생으로 서버 시작 실패: {e}")
# %%
