@echo off
echo 파일 공유 서버를 시작합니다...
echo.

REM Conda 초기화 및 가상환경 활성화
echo Conda 가상환경(general2)을 활성화합니다...

REM Conda 초기화 (Conda 명령어가 배치 파일에서 작동하게 함)
call conda activate base

REM genera2 가상환경 활성화
call conda activate general2

echo 가상환경 활성화 완료!
echo.

echo 서버를 시작합니다...
echo 웹 브라우저에서 접속 주소:
echo http://211.198.13.109:5000/?api_key=[암호화된 키]
echo.
echo 서버를 종료하려면 이 창을 닫으세요.

REM 현재 파일 위치에서 connection.py 실행
python "%~dp0connection2.py"

echo 서버가 종료되었습니다.
pause