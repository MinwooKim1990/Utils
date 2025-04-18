from flask import Flask, request, send_file, abort, render_template, redirect, url_for, session, Response, stream_with_context, jsonify
import os
import datetime
import functools
import secrets
import math
import mimetypes
import re
from stringencode import encode_string, decode_string
from markupsafe import escape
from dotenv import load_dotenv
import google.generativeai as genai
import json
from PIL import Image
import io
import google.api_core.exceptions as exceptions
from google.generativeai.types import BlockedPromptException
from google.generativeai import types
import subprocess
import tempfile
import glob
import time # <<< Add time import for polling
import threading
import uuid # For unique subtitle IDs

# <<< RAG Imports >>>
import fitz # PyMuPDF # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Flask 세션 사용을 위한 secret key

# Load Gemini API Key from environment variable
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    print("경고: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    # Potentially abort or use a default/dummy key for development
    # For now, we'll store None or an empty string
app.config['GEMINI_API_KEY'] = gemini_api_key

# 평문 API 키와 관련 설정
API_KEY = 'MinwooKim1990'
ALLOWED_DRIVES = ['D:/', 'E:/', 'I:/']
HIDDEN_NAMES = {'system volume information', '$recycle.bin'}

# 인증된 디바이스 설정
REQUIRED_DEVICE = '6030'
REQUIRED_BROWSER = 'true'

tree_cache = {}

# --- Gemini API 호출 공통 래퍼 ---
def gemini_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except exceptions.GoogleAPICallError as e:
        print(f"[Gemini API] GoogleAPICallError: {e}")
        return {"error": f"Gemini API 호출 오류: {e.message}", "status": 500}
    except BlockedPromptException as e:
        print(f"[Gemini API] BlockedPromptException: {e}")
        return {"error": f"요청 차단됨: {e}", "status": 403}
    except Exception as e:
        print(f"[Gemini API] Unexpected error: {e}")
        return {"error": f"예상치 못한 오류: {e}", "status": 500}

# --- Redis 캐시 래퍼 및 fallback 구조 ---
try:
    import redis
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_client.ping()
    print("[Cache] Redis 연결 성공. Redis 캐시 사용.")
    USE_REDIS = True
except Exception as e:
    print(f"[Cache] Redis 연결 실패 또는 미설치. In-memory 캐시 사용. 이유: {e}")
    redis_client = None
    USE_REDIS = False

def cache_set(key, value):
    import json
    if USE_REDIS:
        redis_client.set(key, json.dumps(value))
    else:
        in_memory_cache[key] = value

def cache_get(key):
    import json
    if USE_REDIS:
        val = redis_client.get(key)
        if val is not None:
            return json.loads(val)
        return None
    else:
        return in_memory_cache.get(key)

def cache_delete(key):
    if USE_REDIS:
        redis_client.delete(key)
    else:
        in_memory_cache.pop(key, None)

# 기존 in-memory dict 유지 (fallback용)
in_memory_cache = {}

# 기존 dict들 초기화
document_vector_stores = in_memory_cache
video_processing_state = in_memory_cache
subtitle_cache = in_memory_cache

# <<< RAG Initialization: 이전 방식으로 복원 (스크립트 시작 시 로드) >>>
embedding_model = None
embedding_model_load_error_msg = None # 오류 메시지 저장은 유용할 수 있어 유지
try:
    print("Loading Sentence Transformer model (reverting to initial load)...")
    # 모델 이름은 이전과 동일하게 사용
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    embedding_model = None # 실패 시 None으로 설정
    embedding_model_load_error_msg = str(e)
    print(f"FATAL: Failed to load Sentence Transformer model: {e}")
    # 여기서 앱 실행을 중단할 수도 있음 (선택 사항)
    # raise RuntimeError(f"Failed to load embedding model: {e}")

# In-memory store for document chunks and embeddings
# Structure: { instanceId: { 'chunks': [chunk1, chunk2, ...], 'embeddings': [emb1, emb2, ...] } }
document_vector_stores = {}

# <<< Changed video state store >>>
# Structure: { instanceId: { 'rel_path': str, 'file_uri': str, 'file_name_google': str } }
video_processing_state = {}
# <<< End Changed video state store >>>

# <<< NEW: In-memory store for temporary subtitles >>>
# Structure: { subtitle_id: vtt_content }
subtitle_cache = {}
# <<< END NEW subtitle store >>>

def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        import time

        # --- 세션 보안 강화: IP 바인딩 및 타임아웃 ---
        client_ip = request.remote_addr
        now = time.time()
        timeout_seconds = 1800  # 30분

        # IP 체크
        if 'client_ip' in session:
            if session['client_ip'] != client_ip:
                print(f"[Auth] IP 변경 감지: {session['client_ip']} -> {client_ip}")
                session.clear()
                return render_template('login.html', error="IP 변경 감지. 다시 로그인하세요.")
        else:
            session['client_ip'] = client_ip

        # 세션 타임아웃 체크
        last_active = session.get('last_active', now)
        if now - last_active > timeout_seconds:
            print(f"[Auth] 세션 타임아웃 초과: {now - last_active}초 경과")
            session.clear()
            return render_template('login.html', error="세션 만료. 다시 로그인하세요.")
        else:
            session['last_active'] = now
        # --- 세션 보안 강화 끝 ---

        received_key = request.args.get('api_key')
        
        # Device 파라미터 확인 (Safari 관련 제거)
        device = request.args.get('device')
        
        # 세션에 이미 인증됨을 저장하고 있는지 확인
        if 'authenticated' in session and session['authenticated']:
             # 인증된 후에는 device 파라미터 없어도 세션 확인만으로 통과
             if session.get('device') == REQUIRED_DEVICE:
                 return f(*args, **kwargs)
             else:
                 # 세션은 인증되었으나 기기 정보가 없거나 다른 경우 (비정상 상태)
                 session.clear() # 세션 초기화
                 return render_template('login.html', error="기기 인증 정보가 유효하지 않습니다. 다시 로그인해주세요.")

        # Device 파라미터가 처음 제공된 경우 세션에 저장
        if device:
            session['device'] = device
            # 필요한 값과 일치하는지 확인 (여기서 바로 검사)
            if device != REQUIRED_DEVICE:
                 session.clear()
                 return render_template('login.html', error=f"인증되지 않은 기기({device})입니다. 접근이 거부되었습니다.")

        # 세션에 암호가 없고 plain API_KEY로 접속한 경우
        if 'encrypt_password' not in session:
            if received_key == API_KEY:
                # Device 파라미터가 올바른 경우에만 진행 (세션 확인)
                if session.get('device') == REQUIRED_DEVICE:
                    random_password = secrets.token_hex(8)
                    session['encrypt_password'] = random_password
                    session['authenticated'] = True
                    encrypted_api_key = encode_string(API_KEY, random_password)
                    # 리디렉션: file_browser 로, device 파라미터 제거하고 암호화된 키 사용
                    # Linter Fix: Pass args explicitly
                    redirect_args = {
                        'drive': get_base_dir(),
                        'path': request.args.get('path', ''), # Keep current path if provided
                        'api_key': encrypted_api_key
                    }
                    # Linter Fix: Pass args explicitly
                    clean_args = {k: v for k, v in redirect_args.items() if v is not None}
                    return redirect(url_for('file_browser',
                                            drive=clean_args.get('drive'),
                                            path=clean_args.get('path'),
                                            api_key=clean_args.get('api_key')))
                else:
                    # API 키는 맞지만 device 정보가 없거나 틀린 경우
                    return render_template('login.html', error="기기 인증(device=6030 파라미터)이 필요합니다.")
            else:
                # API 키가 제공되었지만 맞지 않는 경우
                if received_key:
                    return render_template('login.html', error="잘못된 API 키입니다.")
                # API 키가 제공되지 않은 경우
                return render_template('login.html')
        else:
            # 세션에 암호가 있는 경우 (이미 한 번 인증 후 리디렉션 된 상태)
            random_password = session['encrypt_password']
            encrypted_api_key = encode_string(API_KEY, random_password)

            # 암호화된 키로 접근한 경우
            if received_key == encrypted_api_key:
                if session.get('device') == REQUIRED_DEVICE:
                    session['authenticated'] = True # 확실히 인증됨
                    return f(*args, **kwargs)
                else:
                    # 암호화 키는 맞지만 기기 정보가 없는 경우 (비정상)
                    session.clear()
                    return render_template('login.html', error="기기 인증 정보가 유효하지 않습니다. 다시 로그인해주세요.")
            # 실수로 다시 plain 키로 접근한 경우 (리디렉션 필요)
            elif received_key == API_KEY:
                 if session.get('device') == REQUIRED_DEVICE:
                     session['authenticated'] = True
                     # Linter Fix: Pass args explicitly
                     redirect_args = {
                         'drive': get_base_dir(),
                         'path': request.args.get('path', ''),
                         'api_key': encrypted_api_key
                     }
                     # Linter Fix: Pass args explicitly
                     clean_args = {k: v for k, v in redirect_args.items() if v is not None}
                     return redirect(url_for('file_browser',
                                             drive=clean_args.get('drive'),
                                             path=clean_args.get('path'),
                                             api_key=clean_args.get('api_key')))
                 else:
                     session.clear()
                     return render_template('login.html', error="기기 인증 정보가 유효하지 않습니다. 다시 로그인해주세요.")
            else:
                 # 암호화된 키도, plain 키도 아닌 경우
                 session.clear()
                 return render_template('login.html', error="잘못된 API 키 또는 인증 세션입니다.")

    return decorated

@app.route('/')
@require_api_key
def file_browser():
    drive = get_base_dir()
    rel_path = request.args.get('path', '')
    # API 키는 데코레이터 통과 시 유효함 (세션 기반 또는 암호화된 키)
    # 템플릿에는 항상 암호화된 키를 전달 (세션에서 생성)
    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
    tree_data = build_tree(drive)
    tree_html = render_tree(tree_data, drive, encrypted_key)
    entries = get_entries(drive, rel_path)
    return render_template('template.html',
                           entries=entries,
                           api_key=encrypted_key, # Always send encrypted key to template
                           current_path=rel_path,
                           tree_html=tree_html,
                           drive=drive,
                           allowed_drives=ALLOWED_DRIVES,
                           sort_param=request.args.get('sort', 'name'),
                           sort_dir=request.args.get('dir', 'asc'),
                           is_search=False)

@app.route('/search')
@require_api_key
def search_files():
    drive = get_base_dir()
    query = request.args.get('query', '')
    rel_path = request.args.get('path', '') # The path to search within
    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
    tree_data = build_tree(drive)
    tree_html = render_tree(tree_data, drive, encrypted_key)
    
    search_results = []
    search_root = os.path.join(drive, rel_path)
    
    if query:
        try:
            for root, dirs, files in os.walk(search_root):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('$') and not d.startswith('~$') and d.lower() not in HIDDEN_NAMES]
                
                for filename in files:
                    if filename.startswith('.') or filename.startswith('$') or filename.startswith('~$') or filename.lower() in HIDDEN_NAMES:
                        continue
                        
                    if query.lower() in filename.lower(): # Case-insensitive search
                        full_file_path = os.path.join(root, filename)
                        file_rel_path = os.path.relpath(full_file_path, drive).replace("\\", "/")
                        
                        try:
                            file_size = os.path.getsize(full_file_path)
                            formatted_size = format_file_size(file_size)
                            mod_timestamp = os.path.getmtime(full_file_path)
                            mod_time_str = datetime.datetime.fromtimestamp(mod_timestamp).strftime("%Y-%m-%d %H:%M")
                            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
                            ext = os.path.splitext(filename)[1].lower()
                            
                            search_results.append({
                                'name': filename,
                                'is_dir': False,
                                'rel_path': file_rel_path,
                                'is_image': ext in image_extensions,
                                'size': file_size,
                                'formatted_size': formatted_size,
                                'mod_time': mod_time_str,
                                'parent_path': os.path.relpath(root, drive).replace("\\", "/") # Add parent path for context
                            })
                        except OSError:
                            # Handle cases where file might be inaccessible
                            continue 
        except FileNotFoundError:
            # Handle case where the search path doesn't exist
            pass
        except Exception as e:
            print(f"Error during search: {e}") # Log other errors
            pass

    return render_template('template.html',
                           entries=search_results, # Pass search results instead of directory entries
                           api_key=encrypted_key,
                           current_path=rel_path, # Keep current path for breadcrumbs/context
                           tree_html=tree_html,
                           drive=drive,
                           allowed_drives=ALLOWED_DRIVES,
                           search_query=query, # Pass the query back to display it
                           is_search=True)

@app.route('/login')
def login():
    return render_template('login.html')

def get_base_dir():
    drive = request.args.get('drive', 'E:/')
    if drive not in ALLOWED_DRIVES:
        drive = 'E:/'
    return drive

def format_file_size(bytes):
    if bytes == 0:
        return '0 Bytes'
    k = 1024
    sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    i = math.floor(math.log(bytes) / math.log(k))
    return f"{(bytes / (k ** i)):.2f} {sizes[i]}"

def build_tree(current_path):
    tree = []
    try:
        with os.scandir(current_path) as it:
            for entry in sorted(it, key=lambda e: (not e.is_dir(), e.name.lower())):
                if entry.name.startswith('.') or entry.name.startswith('$') or entry.name.startswith('~$') or entry.name.lower() in HIDDEN_NAMES:
                    continue
                node = {
                    'name': entry.name,
                    'rel_path': os.path.relpath(os.path.join(current_path, entry.name), current_path).replace("\\", "/"),
                    'is_dir': entry.is_dir(),
                    'children': []
                }
                if entry.is_dir():
                    sub_path = os.path.join(current_path, entry.name)
                    try:
                        with os.scandir(sub_path) as sub_it:
                            children = []
                            for sub_entry in sorted(sub_it, key=lambda e: e.name.lower()):
                                if sub_entry.name.startswith('.') or sub_entry.name.startswith('$') or sub_entry.name.startswith('~$') or sub_entry.name.lower() in HIDDEN_NAMES:
                                    continue
                                if sub_entry.is_dir():
                                    children.append({
                                        'name': sub_entry.name,
                                        'rel_path': os.path.join(node['rel_path'], sub_entry.name).replace("\\", "/"),
                                        'is_dir': True,
                                        'children': []
                                    })
                            node['children'] = children
                    except Exception:
                        pass
                tree.append(node)
    except Exception:
        pass
    return tree


def get_entries(drive, rel_path):
    # Log the received path immediately after getting it from request.args
    print(f"[get_entries] Received drive='{drive}', rel_path='{rel_path}'")
    full_path = os.path.join(drive, rel_path)
    print(f"[get_entries] Calculated full_path: '{full_path}'") # Log the calculated full path

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    entries = []
    try:
        with os.scandir(full_path) as it:
            for entry in it:
                if entry.name.lower() in HIDDEN_NAMES:
                    continue
                if entry.name.startswith('.') or entry.name.startswith('$') or entry.name.startswith('~$'):
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                file_path = os.path.join(full_path, entry.name)
                if not entry.is_dir():
                    file_size = os.path.getsize(file_path)
                    formatted_size = format_file_size(file_size)
                    mod_timestamp = os.path.getmtime(file_path)
                    mod_time_str = datetime.datetime.fromtimestamp(mod_timestamp).strftime("%Y-%m-%d %H:%M")
                else:
                    file_size = 0
                    formatted_size = ""
                    mod_timestamp = 0
                    mod_time_str = ""
                entries.append({
                    'name': entry.name,
                    'is_dir': entry.is_dir(),
                    'rel_path': os.path.relpath(file_path, drive).replace("\\", "/"),
                    'is_image': (ext in image_extensions) and (not entry.is_dir()),
                    'size': file_size,
                    'formatted_size': formatted_size,
                    'mod_time': mod_time_str,
                    'mod_timestamp': mod_timestamp
                })
    except FileNotFoundError:
        print(f"[get_entries] Error: Directory not found at '{full_path}'. Check if path parameter was decoded correctly (e.g., '+' vs space).")
        # Return empty list as before, but log the specific error
    except PermissionError:
        print(f"[get_entries] Error: Permission denied for directory '{full_path}'.")
        # Return empty list
    except Exception as e:
        # Catch any other unexpected errors during scandir
        print(f"[get_entries] Unexpected error scanning directory '{full_path}': {e}")
        # Keep returning empty list for now, but log the error
    
    # Add parent directory entry if not in root
    if rel_path:
        parent_path = os.path.dirname(rel_path).replace("\\", "/")
        # Ensure root path is represented correctly (empty string)
        if parent_path == ".": parent_path = ""
        entries.insert(0, {
            'name': '..',
            'is_dir': True,
            'rel_path': parent_path,
            'is_image': False,
            'size': 0,
            'formatted_size': '',
            'mod_time': '',
            'mod_timestamp': 0 # Put at the beginning regardless of sort?
            # Or assign a very small timestamp to sort first by date asc
        })

    # Get sort parameter and direction
    sort_param = request.args.get('sort', 'name')
    sort_dir = request.args.get('dir', 'asc')  # Default to ascending
    
    # Handle sorting based on parameter and direction
    if sort_param == 'name':
        entries.sort(key=lambda x: x['name'].lower(), reverse=(sort_dir == 'desc'))
    elif sort_param == 'date':
        entries.sort(key=lambda x: x['mod_timestamp'], reverse=(sort_dir == 'desc'))
    elif sort_param == 'size':
        entries.sort(key=lambda x: x['size'], reverse=(sort_dir == 'desc'))
    
    return entries

@app.route('/filelist')
@require_api_key
def file_list():
    drive = get_base_dir()
    rel_path = request.args.get('path', '')
    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
    entries = get_entries(drive, rel_path)
    return render_template('file_list.html',
                           entries=entries,
                           api_key=encrypted_key,
                           current_path=rel_path,
                           drive=drive,
                           sort_param=request.args.get('sort', 'name'),
                           sort_dir=request.args.get('dir', 'asc'))

def render_tree(tree, drive, encrypted_key):
    html = '<ul>'
    for node in tree:
        if node['is_dir']:
            if node.get('children'):
                html += (
                    f'<li>'
                    f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
                    f'<a href="/?drive={drive}&path={node["rel_path"]}&api_key={encrypted_key}" class="folder-link" data-path="{node["rel_path"]}">{node["name"]}</a>'
                    f'<ul style="display: none;">'
                )
                for child in node['children']:
                    if child['is_dir']:
                        html += (
                            f'<li>'
                            f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
                            f'<a href="/?drive={drive}&path={node["rel_path"]+"/"+child["name"]}&api_key={encrypted_key}" class="folder-link" data-path="{node["rel_path"]+"/"+child["name"]}">{child["name"]}</a>'
                            f'<ul style="display: none;"></ul>'
                            f'</li>'
                        )
                    else:
                        html += (
                            f'<li>'
                            f'<a href="/?drive={drive}&path={node["rel_path"]+"/"+child["name"]}&api_key={encrypted_key}" class="file-link">{child["name"]}</a>'
                            f'</li>'
                        )
                html += '</ul></li>'
            else:
                html += (
                    f'<li>'
                    f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
                    f'<a href="/?drive={drive}&path={node["rel_path"]}&api_key={encrypted_key}" class="folder-link" data-path="{node["rel_path"]}">{node["name"]}</a>'
                    f'<ul style="display: none;"></ul>'
                    f'</li>'
                )
        else:
            html += f'<li><a href="/?drive={drive}&path={node["rel_path"]}&api_key={encrypted_key}" class="file-link">{node["name"]}</a></li>'
    html += '</ul>'
    return html

@app.route('/folder_children')
@require_api_key
def folder_children():
    drive = get_base_dir()
    rel_path = request.args.get('path', '')
    full_path = os.path.join(drive, rel_path)
    children = []
    try:
        with os.scandir(full_path) as it:
            for entry in sorted(it, key=lambda e: e.name.lower()):
                if entry.name.startswith('.') or entry.name.startswith('$') or entry.name.startswith('~$') or entry.name.lower() in HIDDEN_NAMES:
                    continue
                if entry.is_dir():
                    children.append({
                        'name': entry.name,
                        'rel_path': os.path.relpath(os.path.join(full_path, entry.name), drive).replace("\\", "/"),
                        'is_dir': True,
                    })
    except Exception as e:
        return str(e), 500
    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
    html = ""
    for child in children:
        html += (
            f'<li>'
            f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
            f'<a href="/?drive={drive}&path={child["rel_path"]}&api_key={encrypted_key}" class="folder-link" data-path="{child["rel_path"]}">{child["name"]}</a>'
            f'<ul style="display: none;"></ul>'
            f'</li>'
        )
    return html

@app.route('/download')
@require_api_key
def download_file():
    drive = request.args.get('drive', 'E:/')
    if drive not in ALLOWED_DRIVES:
        drive = 'E:/'
    base_dir = drive
    # Provide default value '' and ensure rel_path is not None
    rel_path = request.args.get('path', '')
    if not rel_path:
         # Handle case where path is missing or empty
         abort(400, description="File path is required.")

    # 반드시 get_validated_path 사용 (보안 강화)
    full_path = get_validated_path(base_dir, rel_path)
    if os.path.isfile(full_path):
        return send_file(full_path, as_attachment=True)
    else:
        abort(404)

@app.route('/upload', methods=['POST'])
@require_api_key
def upload_file():
    drive = request.args.get('drive', 'E:/')
    if drive not in ALLOWED_DRIVES:
        drive = 'E:/'
    rel_path = request.args.get('path', '')
    
    # 반드시 get_validated_path 사용 (보안 강화)
    save_dir = get_validated_path(drive, rel_path)
    os.makedirs(save_dir, exist_ok=True) # 디렉토리 생성 추가

    if 'file' not in request.files:
        print("Upload error: 'file' not in request.files") # 디버깅 로그 추가
        abort(400, description="파일이 요청에 없습니다.") # 오류 메시지 명확화
    file = request.files['file']
    
    if not file or not file.filename:
        print(f"Upload error: Invalid file or filename. File: {file}, Filename: {file.filename}") # 디버깅 로그 추가
        abort(400, description="잘못된 파일 또는 파일 이름입니다.") # 오류 메시지 명확화

    filename = file.filename # filename 변수 정의

    filename_parts = filename.rsplit('.', 1)
    # Handle case with no extension or only extension
    if len(filename_parts) == 2 and filename_parts[0]:
        new_name = f"{filename_parts[0]}_MAC.{filename_parts[1]}"
    elif len(filename_parts) == 1: # No extension
        new_name = f"{filename}_MAC"
    else: # Edge case, e.g., ".gitignore" -> parts = ['', 'gitignore']
         new_name = f"_MAC.{filename_parts[1]}" # Or handle differently

    save_path = os.path.join(save_dir, new_name) # full_path 대신 save_dir 사용
    print(f"Attempting to save uploaded file to: {save_path}") # 디버깅 로그 추가
    try:
        # 청크 단위 저장 (connection_ver2_fast.py 방식)
        with open(save_path, 'wb') as f:
            chunk_size = 4 * 1024 * 1024 # 4MB chunk size
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        print(f"Successfully saved uploaded file: {save_path}") # 성공 로그 추가
    except Exception as e:
        print(f"Error saving uploaded file to {save_path}: {e}") # 오류 로그 개선
        # Optionally, try to remove partially uploaded file
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"Removed partially uploaded file: {save_path}")
            except OSError as rm_err:
                print(f"Error removing partially uploaded file {save_path}: {rm_err}")
        abort(500, description=f"업로드된 파일 저장 실패: {e}") # 오류 메시지에 상세 정보 포함

    # session['encrypt_password']가 없을 경우 기본값 제공 필요
    encrypt_password = session.get('encrypt_password')
    if not encrypt_password:
        # 암호가 없는 경우 처리 (예: 로그인 페이지로 리디렉션 또는 오류 발생)
        # 여기서는 임시로 빈 문자열 사용하지만, 실제로는 인증 로직과 연계 필요
        print("Warning: encrypt_password not found in session during upload redirect.")
        # 리디렉션 전에 암호가 설정되도록 보장해야 함
        # return redirect(url_for('login')) # 예시: 로그인 페이지로 리디렉션
        # 임시 조치:
        # random_password = secrets.token_hex(8)
        # session['encrypt_password'] = random_password
        # encrypt_password = random_password
        # 이 부분은 require_api_key 데코레이터가 먼저 실행되므로
        # 이 함수에 도달했을 때는 encrypt_password가 있어야 함.
        # 만약 없다면 로직 오류일 수 있으므로 확인 필요.
        # 지금은 오류보다는 경고 후 진행
        encrypted_key = encode_string(API_KEY, '') # 빈 값으로라도 인코딩 시도
    else:
        encrypted_key = encode_string(API_KEY, encrypt_password)

    return redirect(url_for('file_browser', drive=drive, path=rel_path, api_key=encrypted_key))

@app.route('/delete')
@require_api_key
def delete_file():
    drive = request.args.get('drive', 'E:/')
    if drive not in ALLOWED_DRIVES:
        drive = 'E:/'
    base_dir = drive
    # Provide default value '' and ensure rel_path is not None
    rel_path = request.args.get('path', '')
    if not rel_path:
        # Handle case where path is missing or empty
        abort(400, description="File path is required.")
    
    # 반드시 get_validated_path 사용 (보안 강화)
    full_path = get_validated_path(base_dir, rel_path)

    if os.path.isfile(full_path):
        try:
            os.remove(full_path)
        except OSError as e:
            print(f"Error deleting file {full_path}: {e}")
            abort(500, description="Failed to delete file.")

        # Handle dirname for potentially empty rel_path safely
        parent_path = os.path.dirname(rel_path) if rel_path else ''
        encrypted_key = encode_string(API_KEY, session['encrypt_password'])
        return redirect(url_for('file_browser', drive=drive, path=parent_path, api_key=encrypted_key))
    else:
        # Consider if deleting directories should be supported or return a different error
        abort(404, description="File not found.")

# Helper function to get full, validated path and check existence
def get_validated_path(drive, rel_path):
    """Validates the path, ensures it's within the allowed drive, and checks existence."""
    # Basic security: Prevent path traversal attacks like '../'
    if '..' in rel_path:
        abort(400, description="Invalid path component '..'")

    base_path = os.path.abspath(drive)
    # Join and normalize the path
    requested_path_intermediate = os.path.join(base_path, rel_path)
    requested_path = os.path.normpath(requested_path_intermediate)

    # Security check: Ensure the normalized requested path is still within the allowed base drive path
    # Also check if the path exists
    if not requested_path.startswith(base_path) or not os.path.exists(requested_path):
        print(f"Path validation failed or path does not exist: Base='{base_path}', Requested='{requested_path}'")
        abort(404, description="File or directory not found or access denied.")

    return requested_path

# Route for previewing images (inline)
@app.route('/preview_image/<path:rel_path>')
@require_api_key
def preview_image_file(rel_path):
    drive = get_base_dir()
    full_path = get_validated_path(drive, rel_path) # Use validated path
    if os.path.isfile(full_path):
        # Important: as_attachment=False tells the browser to display the file if possible
        return send_file(full_path, as_attachment=False)
    else:
        abort(404)

# Route for streaming video/audio
@app.route('/stream/<path:rel_path>')
@require_api_key
def stream_file(rel_path):
    drive = get_base_dir() # drive 설정 유지
    if drive not in ALLOWED_DRIVES: # ALLOWED_DRIVES 확인 추가
        print(f"Stream error: Invalid drive '{drive}'. Falling back.")
        drive = 'E:/' # Fallback or abort as needed

    full_path = get_validated_path(drive, rel_path) # Use validated path

    if not os.path.isfile(full_path):
        print(f"Stream error: File not found at '{full_path}'")
        abort(404, "요청한 경로가 파일이 아닙니다.") # 메시지 한국어화

    range_header = request.headers.get('Range', None)
    try:
        size = os.path.getsize(full_path)
    except OSError as e:
        print(f"Stream error: Cannot get size for file '{full_path}': {e}")
        abort(500, "파일 크기를 가져올 수 없습니다.")

    # Use optimized chunk size from connection_ver2_fast
    chunk_size = 4 * 1024 * 1024 # 4MB chunks

    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type:
        mime_type = 'application/octet-stream' # Default if type can't be guessed
    print(f"Streaming file: {full_path}, Size: {size}, MIME: {mime_type}, Range: {range_header}") # Log info

    start_byte = 0
    end_byte = size - 1 # Default to entire file
    status_code = 200 # OK by default

    if range_header:
        try:
            # Parse "bytes=start-[end]" or "bytes=start-"
            range_val = range_header.replace('bytes=', '')
            parts = range_val.split('-')
            start_byte = int(parts[0])
            # Handle range end: 'bytes=500-' means from 500 to end
            if len(parts) > 1 and parts[1]:
                end_byte = int(parts[1])
            else:
                # No end byte specified, use file end
                end_byte = size - 1

            # Validate range
            if start_byte >= size or end_byte >= size or start_byte > end_byte:
                 print(f"Stream error: Invalid range {start_byte}-{end_byte} for size {size}")
                 raise ValueError("Invalid range") # Trigger 416 response

            status_code = 206 # Partial Content
            print(f"Stream range requested: bytes {start_byte}-{end_byte}/{size}") # Log range

        except ValueError:
            # If Range header is malformed or invalid, return 416
            print(f"Stream error: Invalid Range header '{range_header}', sending 416")
            abort(416, description="요청한 범위를 만족할 수 없습니다.")

    length = end_byte - start_byte + 1

    # Use the optimized chunk generation from connection_ver2_fast
    def generate():
        try:
            with open(full_path, 'rb') as f:
                f.seek(start_byte)
                remaining = length
                while remaining > 0:
                    # Read the smaller of chunk_size or remaining bytes
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        # End of file reached unexpectedly?
                        print(f"Stream warning: End of file reached while {remaining} bytes were expected.")
                        break
                    yield data
                    remaining -= len(data)
                # print(f"Stream finished for range {start_byte}-{end_byte}") # Optional: log finish
        except Exception as e:
            # Log error during generation
            print(f"Stream error during generation for {full_path}: {e}")
            # Depending on the client, this might be hard to signal back
            # Yielding an error message might break the stream protocol
            # Consider logging only

    resp = Response(stream_with_context(generate()), status_code, mimetype=mime_type)
    resp.headers.add('Content-Length', str(length))
    resp.headers.add('Accept-Ranges', 'bytes') # Indicate range requests are supported

    if status_code == 206: # Add Content-Range header for partial content
        resp.headers.add('Content-Range', f'bytes {start_byte}-{end_byte}/{size}')

    return resp

# Route for previewing text files
@app.route('/preview_text/<path:rel_path>')
@require_api_key
def preview_text_file(rel_path):
    drive = get_base_dir()
    full_path = get_validated_path(drive, rel_path) # Use validated path

    if not os.path.isfile(full_path):
        abort(404, "Requested path is not a file.")

    mime_type, encoding = mimetypes.guess_type(full_path)

    # Define text-based MIME types and common code/text extensions
    text_mimes = {'text/plain', 'text/html', 'text/css', 'application/javascript',
                  'application/json', 'text/markdown', 'application/xml', 'text/csv',
                  'application/x-python-code', 'application/x-sh'} # Added more
    code_extensions = {'.py', '.js', '.html', '.css', '.java', '.c', '.cpp', '.cs', '.php',
                       '.rb', '.go', '.rs', '.swift', '.kt', '.md', '.txt', '.json',
                       '.xml', '.yaml', '.yml', '.ini', '.cfg', '.log', '.sh', '.bat', '.ps1', '.csv'} # Added more

    file_ext = os.path.splitext(full_path)[1].lower()

    is_text_mime = mime_type in text_mimes if mime_type else False
    is_code_ext = file_ext in code_extensions

    # Allow preview if MIME type is text-like OR extension is in our list
    if not is_text_mime and not is_code_ext:
         # Check file size limit before attempting to read
         try:
             if os.path.getsize(full_path) > 10 * 1024 * 1024: # Example: 10MB limit
                 return Response("Preview not available: File is too large.", status=413, mimetype='text/plain')
         except OSError:
             pass # Ignore if size cannot be determined

         # Return a simple plain text message if not previewable and not too large
         return Response(f"Preview not available for this file type ({mime_type or 'unknown type'}).", status=400, mimetype='text/plain')

    try:
        import chardet
    
        max_preview_bytes = 1 * 1024 * 1024  # 1MB 미리보기 제한
    
        # 우선 chardet으로 인코딩 감지
        with open(full_path, 'rb') as f:
            raw = f.read(max_preview_bytes)
            result = chardet.detect(raw)
            detected_encoding = result.get('encoding') or 'utf-8'
            print(f"Detected encoding by chardet: {detected_encoding}")
    
        # 감지된 인코딩으로 최대 1MB까지만 읽기
        try:
            with open(full_path, 'r', encoding=detected_encoding, errors='replace') as f:
                content = f.read(max_preview_bytes)
        except Exception as e:
            print(f"Error reading with detected encoding {detected_encoding}: {e}")
            # 실패 시 바이너리 재시도
            with open(full_path, 'rb') as f:
                raw = f.read(max_preview_bytes)
                content = raw.decode('utf-8', errors='replace')
                detected_encoding = 'utf-8 (fallback)'
    
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>Preview: {escape(os.path.basename(full_path))}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 10px; background-color: var(--preview-bg, #f8f8f8); color: var(--preview-fg, #111); }}
                pre {{
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    background-color: var(--preview-code-bg, #fff);
                    border: 1px solid var(--preview-border, #ddd);
                    padding: 15px;
                    border-radius: 4px;
                    font-family: Consolas, 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    color: var(--preview-code-fg, #111);
                 }}
            </style>
        </head>
        <body>
            <pre>{escape(content)}</pre>
        </body>
        </html>
        """
        return Response(html_content, mimetype='text/html')
    
    except Exception as e:
        print(f"Error previewing text file {full_path}: {e}")
        error_message = f"Error generating file preview for {escape(os.path.basename(full_path))}: {escape(str(e))}"
        return Response(error_message, status=500, mimetype='text/plain')

# Route for previewing PDF files (inline using browser viewer)
@app.route('/preview_pdf/<path:rel_path>')
@require_api_key
def preview_pdf_file(rel_path):
    drive = get_base_dir()
    full_path = get_validated_path(drive, rel_path) # Use validated path
    if os.path.isfile(full_path):
        # Send the file directly, browser should use its PDF viewer
        return send_file(full_path, as_attachment=False)
    else:
        abort(404)

# --- RAG Helper Functions --- START

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        print(f"Extracted {len(text)} characters from PDF: {os.path.basename(file_path)}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return None

def extract_text_from_plain(file_path):
    """Extracts text from plain text files (txt, md, py, etc.). Tries common encodings."""
    encs = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
    for enc in encs:
        try:
            with open(file_path, 'r', encoding=enc) as f: text = f.read()
            print(f"Extracted {len(text)} chars from plain text: {os.path.basename(file_path)} using {enc}")
            return text
        except UnicodeDecodeError: continue
        except Exception as e: print(f"Error plain extract {file_path} with {enc}: {e}"); return None
    try: # Fallback binary read
        with open(file_path, 'rb') as f: b_content = f.read()
        text = b_content.decode('utf-8', errors='replace')
        print(f"Extracted {len(text)} chars plain (binary fallback): {os.path.basename(file_path)}")
        return text
    except Exception as e: print(f"Final fallback plain error {file_path}: {e}"); return None

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks with overlap (simple character-based)."""
    if not text: return []
    chunks = []; start = 0
    while start < len(text):
        end = start + chunk_size; chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): break
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

def get_embeddings(texts):
    """Generates embeddings for a list of texts using the loaded model."""
    if embedding_model is None or not texts:
        print("Warning/Error: get_embeddings called but model is not loaded or texts empty.")
        return None
    try:
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        print(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Finds top_k most relevant chunks based on cosine similarity."""
    if query_embedding is None or chunk_embeddings is None or not chunks: return []
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1)[:, np.newaxis]
    similarities = np.dot(chunk_norms, query_norm)
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_k]
    relevant_chunks = [(chunks[i], similarities[i]) for i in top_indices]
    print(f"Found {len(relevant_chunks)} relevant chunks with similarities: {[f'{s:.3f}' for _, s in relevant_chunks]}")
    return [chunk for chunk, _ in relevant_chunks]

# --- RAG Helper Functions --- END

# --- Document Processing Endpoint --- START
@app.route('/process_document', methods=['POST'])
@require_api_key
def process_document():
    # <<< 모델 로딩 상태 확인 (시작 시 로드 실패했는지) >>>
    if embedding_model is None:
        print("Error: /process_document called but embedding model failed to load initially.")
        return jsonify({"error": f"문서 처리 모델 초기 로드 실패: {embedding_model_load_error_msg}"}), 500

    # <<< Lazy Loading 관련 로직 제거 >>>
    # if embedding_model_status == "not_loaded": ...
    # elif embedding_model_status == "loading": ...
    # elif embedding_model_status == "error": ...

    # <<< 이하 로직은 모델 로드 가정 하에 진행 >>>
    if not request.content_type or not request.content_type.startswith('application/json'):
        return jsonify({"error": "Request must be JSON."}), 415

    # Parse JSON data from the request
    data = request.json
    if data is None:
        return jsonify({"error": "Invalid JSON data received."}), 400

    # Extract required fields from JSON data
    rel_path = data.get('relPath')
    instance_id = data.get('instanceId')
    file_name = data.get('fileName')
    drive = data.get('drive')

    # <<< drive 관련 검증 코드도 그대로 두어야 합니다 >>>
    if not rel_path or not instance_id or not file_name or not drive:
         return jsonify({"error": "Missing required parameters (relPath, instanceId, fileName, drive)."}), 400
    if drive not in ALLOWED_DRIVES:
         print(f"Error: Invalid drive '{drive}' received in /process_document request.")
         return jsonify({"error": f"Invalid drive specified: {drive}"}), 400
    # drive = get_base_dir() # <<< 이 줄은 주석 처리하거나 없어야 합니다.

    try:
        full_path = get_validated_path(drive, rel_path) # <<< 여기서 전달받은 drive 사용 확인
        
        file_ext = os.path.splitext(file_name)[1].lower()
        print(f"Processing document (original method): {rel_path} for instance: {instance_id}")

        text = None
        supported_exts = ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.log', '.sh', '.bat', '.ps1']
        if file_ext == '.pdf': text = extract_text_from_pdf(full_path)
        elif file_ext in supported_exts: text = extract_text_from_plain(full_path)
        else:
            print(f"Error: Unsupported file type for RAG: {file_ext}")
            return jsonify({"error": f"지원되지 않는 파일 형식입니다: {file_ext}"}), 415

        if text is None:
            print(f"Error: Failed to extract text from {rel_path}")
            return jsonify({"error": "파일에서 텍스트를 추출하지 못했습니다."}), 500

        chunks = chunk_text(text)
        if not chunks:
            print(f"Error: Document empty or could not be chunked: {rel_path}")
            return jsonify({"error": "문서가 비어있거나 내용을 나눌 수 없습니다."}), 400

        embeddings = get_embeddings(chunks) # <<< 직접 호출
        if embeddings is None:
            # 이 경우는 모델 로드 성공 후 임베딩 생성 실패
            print(f"Error: Failed to generate embeddings for {rel_path}")
            return jsonify({"error": "텍스트 임베딩 생성에 실패했습니다."}), 500

        document_vector_stores[instance_id] = {
            'chunks': chunks,
            'embeddings': embeddings,
            'rel_path': rel_path
        }
        print(f"Successfully processed document (original method) for instance {instance_id}.")
        return jsonify({"message": f"문서 처리 완료: {len(chunks)}개 청크 생성됨"}), 200

    except FileNotFoundError:
        print(f"Error: File not found during document processing: {rel_path}")
        # <<< JSON 응답 반환 >>>
        return jsonify({"error": f"파일을 찾을 수 없습니다: {rel_path}"}), 404
    except PermissionError:
        print(f"Error: Permission denied for file during document processing: {rel_path}")
        # <<< JSON 응답 반환 >>>
        return jsonify({"error": f"파일 접근 권한이 없습니다: {rel_path}"}), 403
    except Exception as e:
        print(f"Error processing document {rel_path} for instance {instance_id}: {e}")
        import traceback
        traceback.print_exc() # Log the full traceback for debugging
        # <<< JSON 응답 반환 >>>
        return jsonify({"error": f"문서 처리 중 예상치 못한 오류 발생: {e}"}), 500

# --- Document Processing Endpoint --- END

# --- Video Processing Helper Functions --- START

# --- Video Processing Helper Functions --- END


# --- Video Processing Endpoints --- START

@app.route('/process_video', methods=['POST'])
@require_api_key
def process_video():
    # Check API Key configuration first
    api_key = app.config.get('GEMINI_API_KEY')
    if not api_key:
        print("[process_video] Error: Gemini API key not configured.") # Debug log
        return jsonify({"error": "Gemini API key not configured."}), 500

    # Check Content-Type before accessing request.json
    if not request.content_type or not request.content_type.startswith('application/json'):
        print("[process_video] Error: Request must be JSON.") # Debug log
        return jsonify({"error": "Request must be JSON."}), 415

    data = request.json
    # Add check: Ensure data is not None
    if data is None:
        print("[process_video] Error: Invalid JSON data received.") # Debug log
        return jsonify({"error": "Invalid JSON data received."}), 400

    rel_path = data.get('relPath')
    instance_id = data.get('instanceId')
    file_name_original = data.get('fileName') # Get original filename for display name
    drive = data.get('drive') # <<< Get drive from JSON payload >>>

    if not rel_path or not instance_id or not file_name_original or not drive:
        print(f"[process_video] Error: Missing required parameters. Received: relPath={rel_path}, instanceId={instance_id}, fileName={file_name_original}, drive={drive}") # Debug log
        return jsonify({"error": "Missing required parameters (relPath, instanceId, fileName, drive)."}), 400

    # <<< Validate the received drive against allowed drives >>>
    if drive not in ALLOWED_DRIVES:
         print(f"[process_video] Error: Invalid drive '{drive}' received.") # Debug log
         return jsonify({"error": f"Invalid drive specified: {drive}"}), 400

    # <<< Use the drive from payload for validation >>>
    try:
        full_path = get_validated_path(drive, rel_path)
    except Exception as validation_err:
        print(f"[process_video] Error validating path: {validation_err}") # Debug log
        # Abort or return jsonify based on get_validated_path behavior (it aborts)
        # If get_validated_path aborts, this might not be reached directly, but good practice
        return jsonify({"error": f"Path validation failed: {validation_err}"}), 404 # Or appropriate code

    print(f"[process_video] Initiating Gemini File API upload for: {rel_path} (Instance: {instance_id}, Drive: {drive}, Full Path: {full_path})") # Debug log

    try:
        genai.configure(api_key=api_key)
        # === Gemini File API Upload ===
        print("[process_video] Uploading video file to Google...") # Debug log
        # Use the original filename as the display name for the uploaded file
        video_file = genai.upload_file(path=full_path, display_name=file_name_original) # type: ignore
        print(f"[process_video] Upload started: URI={video_file.uri}, Name={video_file.name}") # Debug log

        # === Polling for Processing State ===
        print("[process_video] Waiting for file processing...", end='') # Debug log
        polling_start_time = time.time()
        max_polling_time = 300 # 5 minutes timeout for polling
        while video_file.state.name == "PROCESSING":
            if time.time() - polling_start_time > max_polling_time:
                print("\n[process_video] Error: Polling timeout reached.") # Debug log
                # Attempt to delete the potentially stuck file
                try: genai.delete_file(name=video_file.name); print(f"[process_video] Cleaned up file {video_file.name} after timeout.") # type: ignore
                except Exception as del_err: print(f"[process_video] Warning: Failed to cleanup {video_file.name} after timeout: {del_err}")
                return jsonify({"error": "File processing timed out."}), 500

            print('.', end='')
            time.sleep(5) # Increase polling interval slightly to 5 seconds
            try:
                video_file = genai.get_file(name=video_file.name) # type: ignore
                print(f" PState: {video_file.state.name}", end='') # Log current state during poll
            except Exception as get_err:
                 print(f"\n[process_video] Error getting file status during polling: {get_err}") # Debug log
                 # Decide how to handle polling error (e.g., retry, abort)
                 # For now, aborting
                 return jsonify({"error": f"Error checking file status: {get_err}"}), 500

        print(' Done.')

        if video_file.state.name == "FAILED":
            print(f"[process_video] File processing failed: {video_file.name}") # Debug log
            # Attempt to delete the failed file from Google
            try:
                genai.delete_file(name=video_file.name) # type: ignore
                print(f"[process_video] Cleaned up failed file: {video_file.name}")
            except Exception as delete_err:
                print(f"[process_video] Warning: Failed to clean up failed file {video_file.name}: {delete_err}")
            return jsonify({"error": "File processing failed on Google's side."}), 500
        elif video_file.state.name != "ACTIVE":
             print(f"[process_video] File ended in unexpected state: {video_file.state.name}") # Debug log
             return jsonify({"error": f"File processing ended in unexpected state: {video_file.state.name}"}), 500

        # Store URI and Google's file name
        video_processing_state[instance_id] = {
            'rel_path': rel_path, # Keep local path for reference if needed
            'file_uri': video_file.uri,
            'file_name_google': video_file.name # Store Google's internal name for deletion
        }
        print(f"[process_video] Video processing complete and state stored for instance {instance_id}: {video_processing_state[instance_id]}") # Debug log

        return jsonify({
            "message": f"영상 파일 업로드 및 처리 완료: {video_file.display_name}",
            "fileUri": video_file.uri
        }), 200

    except FileNotFoundError:
        print(f"[process_video] Error: Local video file not found at {full_path}") # Debug log
        return jsonify({"error": "Local video file not found."}), 404
    except exceptions.GoogleAPICallError as e:
         print(f"[process_video] Gemini API Call Error: {e}") # Debug log
         return jsonify({"error": f"Gemini API Error: {e.message}"}), 500
    except Exception as e:
        print(f"[process_video] Unexpected error: {e}") # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred during video processing."}), 500

# --- Video Processing Endpoints --- END

# --- Instance Cache Clearing Endpoint --- START
@app.route('/clear_instance_cache', methods=['POST'])
@require_api_key
def clear_instance_cache():
    # Check Content-Type before accessing request.json
    if not request.content_type or not request.content_type.startswith('application/json'):
        return jsonify({"error": "Request must be JSON."}), 415

    data = request.json
    # Add check: Ensure data is not None
    if data is None:
        return jsonify({"error": "Invalid JSON data received."}), 400

    instance_id = data.get('instanceId')

    if not instance_id:
        return jsonify({"error": "Missing instanceId parameter."}), 400

    cleared_doc = False
    cleared_video = False
    google_file_deleted = False

    # Clear from document store
    if instance_id in document_vector_stores:
        del document_vector_stores[instance_id]
        cleared_doc = True
        print(f"Cleared document cache for instance: {instance_id}")

    # Clear from video store and delete from Google
    if instance_id in video_processing_state:
        state = video_processing_state[instance_id]
        file_name_google = state.get('file_name_google')
        del video_processing_state[instance_id]
        cleared_video = True
        print(f"Cleared local video processing state for instance: {instance_id}")

        if file_name_google:
            api_key = app.config.get('GEMINI_API_KEY')
            if api_key:
                try:
                    print(f"Attempting to delete Google file: {file_name_google}")
                    genai.configure(api_key=api_key)
                    genai.delete_file(name=file_name_google) # type: ignore
                    google_file_deleted = True
                    print(f"Successfully deleted Google file: {file_name_google}")
                except exceptions.NotFound:
                     print(f"Google file {file_name_google} not found for deletion (already deleted?).")
                     google_file_deleted = True # Consider it deleted if not found
                except exceptions.GoogleAPICallError as e:
                     print(f"Error deleting Google file {file_name_google}: {e}")
                     # Decide if this should be a failure or just a warning
                except Exception as e:
                     print(f"Unexpected error deleting Google file {file_name_google}: {e}")
            else:
                print(f"Cannot delete Google file {file_name_google}: API key not configured.")
        else:
            print(f"No Google file name found in state for instance {instance_id}, skipping deletion.")

    if cleared_doc or cleared_video:
        # import gc
        # gc.collect() # Optional: trigger garbage collection
        return jsonify({
            "message": f"Cache cleared for instance {instance_id} (Doc: {cleared_doc}, Video: {cleared_video}, Google File Deleted: {google_file_deleted})"
        }), 200
    else:
        print(f"Attempted to clear cache for non-existent instance: {instance_id}")
        return jsonify({"error": "Instance ID not found in any cache."}), 404

# --- Instance Cache Clearing Endpoint --- END

# --- Gemini Chat Endpoint --- START
@app.route('/chat', methods=['POST'])
@require_api_key
def chat():
    # <<< 모델 상태 변수 불필요 >>>
    # global embedding_model_status, embedding_model

    # ... (요청 파싱 로직 동일) ...
    if not app.config.get('GEMINI_API_KEY'):
        print("[chat] Error: API key not configured.") # Debug log
        return jsonify({"error": "API key not configured."}), 500

    content_type = request.content_type
    data = None; image_file = None; instance_id = None
    if content_type.startswith('application/json'):
        data = request.json
        if data is None: print("[chat] Error: Invalid JSON data received."); return jsonify({"error": "Invalid JSON data received."}), 400 # Debug log
        instance_id = data.get('instanceId')
    elif content_type.startswith('multipart/form-data'):
        data = request.form.to_dict(); image_file = request.files.get('image'); instance_id = data.get('instanceId')
        if 'history' in data:
            try: data['history'] = json.loads(data['history'])
            except json.JSONDecodeError: print("[chat] Error: Invalid history format."); return jsonify({"error": "Invalid history format."}), 400 # Debug log
    else: print(f"[chat] Error: Unsupported Content-Type: {content_type}"); return jsonify({"error": "Unsupported Content-Type"}), 415 # Debug log

    if not data: print("[chat] Error: Invalid request data."); return jsonify({"error": "Invalid request data"}), 400 # Debug log

    message = data.get('message', ''); model_name = data.get('model', 'gemini-1.5-flash-latest'); history = data.get('history', [])
    if not isinstance(history, list): history = []
    print(f"[chat] Received request. Model: {model_name}, InstanceID: {instance_id}, History: {len(history)} msgs, Image: {image_file is not None}") # Debug log

    # <<< RAG 로직 복원: 모델 로딩 상태 체크 제거 >>>
    rag_context = ""
    is_document_chat = False
    if instance_id and instance_id in document_vector_stores:
        is_document_chat = True
        # <<< 모델이 로드되었는지 여부만 체크 (시작 시 로드 실패 대비) >>>
        if embedding_model is not None:
            print(f"[chat] Instance {instance_id} found in doc store, applying RAG.") # Debug log
            doc_store = document_vector_stores[instance_id]
            if message:
                query_embedding = get_embeddings([message])
                if query_embedding is not None:
                    relevant_chunks = find_relevant_chunks(
                        query_embedding[0],
                        doc_store['embeddings'],
                        doc_store['chunks'],
                        top_k=3
                    )
                    if relevant_chunks:
                        rag_context = "\n\n-- 문서 내용 --\n" + "\n\n".join(relevant_chunks) + "\n--------------\n"
                        print(f"[chat] Added RAG context for instance {instance_id}") # Debug log
                    else: print("[chat] No relevant RAG chunks found.") # Debug log
                else: print("[chat] Failed to generate query embedding for RAG.") # Debug log
            else: print("[chat] Skipping RAG for empty message.") # Debug log
        else:
             print(f"[chat] Skipping RAG for instance {instance_id}, embedding model not available.") # Debug log


    # ... (이하 /chat 엔드포인트의 나머지 로직 동일: 비디오 처리, 프롬프트 준비, API 호출, 스트리밍 등) ...
    is_video_chat = False; video_file_object = None
    if instance_id and instance_id in video_processing_state and not is_document_chat:
        is_video_chat = True; video_state = video_processing_state[instance_id]; file_uri = video_state.get('file_uri')
        file_name_google = video_state.get('file_name_google') # <<< Get Google's file name
        print(f"[chat] Instance {instance_id} found in video store. URI: {file_uri}, Google Name: {file_name_google}") # Debug log
        if file_name_google: # <<< Check if we have the Google file name
            try:
                print(f"[chat] Attempting to get File object using name: {file_name_google}")
                video_file_object = genai.get_file(name=file_name_google) # type: ignore
                print(f"[chat] Successfully retrieved video File object for instance {instance_id}. State: {video_file_object.state.name}") # Debug log
            except exceptions.NotFound:
                print(f"[chat] Error retrieving video File object: File {file_name_google} not found on Google.")
                video_file_object = None
            except Exception as e:
                print(f"[chat] Error retrieving video File object for instance {instance_id} using name {file_name_google}: {e}")
                video_file_object = None
        else: print(f"[chat] Warning: Google file name not found in state for instance {instance_id}") # Debug log

    # <<< Corrected try-except block for model initialization >>>
    try:
        genai.configure(api_key=app.config['GEMINI_API_KEY'])
        model = genai.GenerativeModel(model_name)
        print(f"[chat] Gemini model '{model_name}' initialized.") # Debug log
    except Exception as e: # <<< Corrected indentation
        print(f"[chat] Error initializing model '{model_name}': {e}") # Debug log
        return jsonify({"error": f"Failed to initialize model: {e}"}), 500

    chat_session = model.start_chat(history=history); final_prompt_text = ""
    if rag_context: final_prompt_text += rag_context
    if message: final_prompt_text += message

    image_prompt_part = None
    if image_file and not is_document_chat:
        print(f"[chat] Processing attached image: {image_file.filename} ({image_file.mimetype})") # Debug log
        try:
            img_bytes = image_file.read()
            # <<< 수정: types.Part.from_bytes 사용 >>>
            image_prompt_part = {
                "mime_type": image_file.mimetype,
                "data": img_bytes
            }
            print(f"[chat] Image bytes read and converted to Part: {len(img_bytes)}") # Debug log
            # Check if history is empty and prepend initial analysis prompt for images
            # Note: This logic might need adjustment depending on desired UX
            if not history or len(history) == 0:
                initial_prompt = ("이 이미지를 자세히 분석하고 주요 요소, 객체, 장면, 분위기 등을 설명해주세요. "
                                "가능하다면 텍스트(OCR)도 추출해주세요. "
                                "분석이 끝나면 이 이미지에 대해 무엇이 궁금한지 물어봐주세요.")
                final_prompt_text = initial_prompt + ("\n\n" + final_prompt_text if final_prompt_text else "")
                print("[chat] Prepended initial image analysis prompt.") # Debug log
        except Exception as e: print(f"[chat] Error processing image file: {e}"); image_prompt_part = None # Debug log

    final_api_parts = []

    # <<< MODIFICATION START: Prioritize media parts >>>
    # Add VIDEO file object first if it's a video chat
    if video_file_object and is_video_chat:
         final_api_parts.append(video_file_object)
         print(f"[chat] Added video part FIRST to final API parts.") # Log change

    # Add IMAGE part if present (independent of video)
    if image_prompt_part: 
         final_api_parts.append(image_prompt_part)
         print(f"[chat] Added image part to final API parts.") # Adjusted log

    # Add text part AFTER media parts
    if final_prompt_text:
        final_api_parts.append(final_prompt_text)
        print(f"[chat] Added text part AFTER media to final API parts.") # Log change
    # <<< MODIFICATION END >>>

    # <<< DEBUG LOGGING for final_api_parts >>>
    print(f"[chat] Final API Parts prepared ({len(final_api_parts)} parts):") # Debug log
    for i, part in enumerate(final_api_parts):
        if isinstance(part, str):
            print(f"  Part {i}: String (length={len(part)}) - Preview: '{part[:100]}...'" if len(part) > 100 else f"  Part {i}: String - '{part}'") # Debug log
        elif isinstance(part, dict) and 'mime_type' in part and 'data' in part:
            print(f"  Part {i}: Image/Data - MimeType: {part['mime_type']}, Data Length: {len(part['data'])}") # Debug log
        elif hasattr(part, 'uri'): # Assuming it's a genai.Part object with a URI
             # <<< Safely access uri using getattr >>>
             part_uri = getattr(part, 'uri', None)
             print(f"  Part {i}: URI Part - URI: {part_uri}") # Debug log
        else: # <<< Corrected indentation
            print(f"  Part {i}: Unknown type - {type(part)}") # Debug log
    # <<< END DEBUG LOGGING >>>

    if not final_api_parts:
        print("[chat] No content to send, returning empty message.") # Debug log
        def empty_stream(): yield f'event: data\ndata: {json.dumps({"response": "메시지를 입력해주세요."})}\n\n'; yield f'event: end\ndata: {{}}\n\n'
        return Response(empty_stream(), mimetype='text/event-stream')

    print(f"[chat] Sending {len(final_api_parts)} parts to Gemini model {model_name}...") # Debug log

    def stream():
        nonlocal model, chat_session # Allow modification of outer scope variables
        try:
            # <<< MODIFICATION START: Check model compatibility for video >>>
            effective_model_name = model_name
            if is_video_chat and not model_name.startswith("gemini-1.5"):
                 # Force a video-capable model if needed
                 effective_model_name = "gemini-1.5-flash-latest" # Or gemini-1.5-pro-latest
                 print(f"[chat] Warning: Original model '{model_name}' might not support video. Switching to '{effective_model_name}'.")
                 # Re-initialize model and chat session with the compatible model
                 try:
                     model = genai.GenerativeModel(effective_model_name)
                     chat_session = model.start_chat(history=history) # Use original history
                     print(f"[chat] Re-initialized model and chat session for video with {effective_model_name}.")
                 except Exception as model_init_err:
                      print(f"[chat] CRITICAL: Failed to re-initialize model to {effective_model_name}: {model_init_err}")
                      # Fallback or error handling needed here. Yielding error message.
                      yield f'data: {json.dumps({"error": f"비디오 호환 모델({effective_model_name}) 초기화 실패: {model_init_err}"})}\n\n'
                      yield f'event: end\ndata: Error occurred\n\n'
                      return # Stop the stream generator
            # <<< MODIFICATION END >>>

            # Use the potentially updated chat_session
            response_stream = chat_session.send_message(final_api_parts, stream=True)
            print(f"[chat] Stream initiated using model {effective_model_name}.") # Debug log
            for chunk in response_stream:
                if chunk.parts:
                     # print(f"[chat] Stream chunk received: {chunk.parts[0].text[:50]}...") # Verbose: log each chunk
                     yield f'data: {json.dumps({"response": chunk.parts[0].text})}\n\n'
                else:
                     print("[chat] Stream chunk received with no parts.") # Debug log
            print("[chat] Stream finished.") # Debug log
            yield f'event: end\ndata: Stream ended\n\n'
        except exceptions.GoogleAPICallError as e:
            error_message = f"API 호출 오류: {e.message}"
            print(f"[chat] GoogleAPICallError: {error_message}") # Debug log
            if "503" in error_message or "overload" in error_message.lower(): error_message = "모델 사용량 초과(503)."
            yield f'data: {json.dumps({"error": error_message})}\n\n'; yield f'event: end\ndata: Error occurred\n\n'
        except BlockedPromptException as e:
            print(f"[chat] BlockedPromptException: {e}") # Debug log
            yield f'data: {json.dumps({"error": f"요청 차단됨: {e}"})}\n\n'; yield f'event: end\ndata: Error occurred\n\n'
        # <<< Corrected indentation for the general exception >>>
        except Exception as e:
            print(f"[chat] Exception during streaming: {e}") # Debug log
            import traceback
            traceback.print_exc() # Log full traceback for unexpected errors
            yield f'data: {json.dumps({"error": f"스트리밍 중 예상치 못한 오류 발생: {e}"})}\n\n'; yield f'event: end\ndata: Error occurred\n\n'

    return Response(stream(), mimetype='text/event-stream')

# --- Gemini Chat Endpoint --- END

# --- Gemini Audio Chat Endpoint --- START
@app.route('/chat_audio', methods=['POST'])
@require_api_key # Ensure the user is authenticated
def handle_audio_chat():
    api_key = app.config.get('GEMINI_API_KEY')
    if not api_key:
        return {"error": "Gemini API 키가 설정되지 않았습니다."}, 500

    # Configure the Gemini client
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Gemini 구성 오류: {e}")
        return {"error": "Gemini 클라이언트 설정에 실패했습니다."}, 500

    # Check if audio file is present
    if 'audio' not in request.files:
        return {"error": "오디오 파일이 요청에 포함되지 않았습니다."}, 400

    audio_file = request.files['audio']
    model_name_req = request.form.get('model', 'gemini-1.5-flash-latest') # Get model from form or default

    # Force an audio-capable model (adjust if needed)
    # Note: Check Gemini documentation for the latest recommended model for audio
    model_name = "gemini-1.5-pro-latest" # Or another suitable model
    print(f"Audio detected. Forcing model to {model_name} (original request: {model_name_req})")

    try:
        # Prepare audio part
        # Gemini API can often handle common web audio formats directly
        audio_bytes = audio_file.read()
        audio_part = {
            "mime_type": audio_file.mimetype, # Use the mimetype sent by browser
            "data": audio_bytes
        }
        print(f"[Debug] Prepared audio_part: mime_type={audio_part['mime_type']}, data_length={len(audio_part['data'])}")

        # --- API Call --- #
        model = genai.GenerativeModel(model_name)

        # Simple prompt for audio - enhance as needed
        prompt = "이 오디오 파일의 내용을 설명해주세요." 
        contents = [prompt, audio_part]

        print(f"Sending audio to Gemini ({model_name})...")
        response = model.generate_content(contents)

        # --- Handle Response --- #
        if response and response.text:
            print(f"Gemini audio response: {response.text[:100]}...")
            return {"response": response.text}
        else:
             safety_feedback = getattr(response, 'prompt_feedback', None)
             block_reason = getattr(safety_feedback, 'block_reason', None)
             if block_reason:
                 print(f"Gemini 응답 차단됨 (오디오): {block_reason}")
                 return {"response": f"(응답이 차단되었습니다: {block_reason})"}
             else:
                 print(f"Gemini 로부터 빈 응답 수신 (오디오): {response}")
                 return {"response": "(모델로부터 오디오 응답을 받지 못했습니다.)"}

    except Exception as e:
        print(f"Gemini API 오디오 호출 오류 ({model_name}): {e}")
        error_detail = str(e)
        # Add specific error checks if possible
        error_msg = f"Gemini API ({model_name}) 오디오 처리 중 오류 발생: {error_detail}"
        return {"error": error_msg}, 500
# --- Gemini Audio Chat Endpoint --- END

# --- >>> NEW: SRT to VTT Conversion Helper <<< ---
def srt_to_vtt(srt_content):
    """Converts SRT formatted string to VTT formatted string."""
    if not srt_content:
        return None
    try:
        vtt_content = "WEBVTT\n\n"
        # Replace comma decimal separator with period for VTT timestamps
        srt_content = srt_content.replace(',', '.')
        # Split SRT into blocks (index, time, text)
        blocks = srt_content.strip().split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                # Skip the index line (usually the first line)
                # Add the timestamp line directly (already modified commas)
                vtt_content += lines[1] + "\n"
                # Add the text lines
                vtt_content += "\n".join(lines[2:]) + "\n\n"
            elif len(lines) == 2: # Handle blocks with only timestamp and one line of text
                vtt_content += lines[0] + "\n" # Assume line 0 is timestamp if only 2 lines
                vtt_content += lines[1] + "\n\n"

        return vtt_content.strip()
    except Exception as e:
        print(f"Error converting SRT to VTT: {e}")
        return None
# --- >>> END SRT to VTT Conversion Helper <<< ---

# --- >>> NEW: Video Info Endpoint <<< ---
@app.route('/video_info/<path:rel_path>')
@require_api_key
def get_video_info(rel_path):
    drive = get_base_dir()
    try:
        video_full_path = get_validated_path(drive, rel_path)
        video_dir = os.path.dirname(video_full_path)
        video_filename_no_ext = os.path.splitext(os.path.basename(video_full_path))[0]

        subtitle_vtt_url = None
        srt_file_path = os.path.join(video_dir, f"{video_filename_no_ext}.srt")

        if os.path.exists(srt_file_path):
            print(f"Found SRT file: {srt_file_path}")
            try:
                # Try reading with common encodings
                srt_content = None
                for enc in ['utf-8', 'cp949', 'euc-kr']:
                    try:
                        with open(srt_file_path, 'r', encoding=enc) as f:
                            srt_content = f.read()
                        print(f"Read SRT with encoding: {enc}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as read_err:
                         print(f"Error reading SRT {srt_file_path} with {enc}: {read_err}")
                         continue
                
                if srt_content is None: # Try binary read as fallback
                    try:
                        with open(srt_file_path, 'rb') as f:
                             srt_content = f.read().decode('utf-8', errors='replace')
                        print("Read SRT with binary fallback (utf-8 replace)")
                    except Exception as bin_read_err:
                        print(f"Failed final binary read for SRT {srt_file_path}: {bin_read_err}")


                if srt_content:
                    vtt_content = srt_to_vtt(srt_content)
                    if vtt_content:
                        subtitle_id = str(uuid.uuid4())
                        subtitle_cache[subtitle_id] = vtt_content
                        encrypted_key = encode_string(API_KEY, session['encrypt_password'])
                        # Construct the URL for the /subtitle endpoint
                        subtitle_vtt_url = url_for('serve_subtitle', subtitle_id=subtitle_id, api_key=encrypted_key, _external=False) # Use relative URL
                        print(f"Generated VTT URL: {subtitle_vtt_url}")
                    else:
                         print("Failed to convert SRT to VTT.")
                else:
                     print(f"Could not read content from SRT file: {srt_file_path}")

            except Exception as e:
                print(f"Error processing SRT file {srt_file_path}: {e}")

        encrypted_key = encode_string(API_KEY, session['encrypt_password'])
        video_stream_url = url_for('stream_file', rel_path=rel_path, drive=drive, api_key=encrypted_key, _external=False) # Use relative URL

        return jsonify({
            "video_url": video_stream_url,
            "subtitle_url": subtitle_vtt_url # Will be null if no subtitle found/processed
        })

    except FileNotFoundError:
        abort(404, "Video file not found.")
    except Exception as e:
        print(f"Error in /video_info for {rel_path}: {e}")
        abort(500, "Error retrieving video information.")
# --- >>> END Video Info Endpoint <<< ---

# --- >>> NEW: Subtitle Serving Endpoint <<< ---
@app.route('/subtitle/<subtitle_id>')
@require_api_key # Reuse API key check for security
def serve_subtitle(subtitle_id):
    vtt_content = subtitle_cache.get(subtitle_id)
    if vtt_content:
        # Optionally remove from cache after serving? For now, keep it.
        # del subtitle_cache[subtitle_id]
        return Response(vtt_content, mimetype='text/vtt; charset=utf-8')
    else:
        print(f"Subtitle ID not found in cache: {subtitle_id}")
        abort(404, "Subtitle not found.")
# --- >>> END Subtitle Serving Endpoint <<< ---

# ----------- Gunicorn + gevent 권장 -----------
# gunicorn -w 4 -k gevent -b 0.0.0.0:443 --certfile=cert.pem --keyfile=key.pem Utils.connection_ver3_upgrade:app

# <<< 앱 실행 부분 복원: 스레드 로직 제거 >>>
if __name__ == '__main__':
    print("Starting Flask server (initial model load attempted).")
    try:
        # cert_path, key_path 정의 부분은 그대로 두거나 주석 처리해도 됩니다.
        cert_path = os.path.join(os.path.dirname(__file__), 'cert.pem')
        key_path = os.path.join(os.path.dirname(__file__), 'key.pem')
        # ssl_context 값을 원래 인증서 경로로 복원하고 포트를 443으로 유지
        print(f"Attempting to run with provided SSL context ({cert_path}, {key_path})...")
        app.run(host='0.0.0.0', port=443, debug=True, ssl_context=(cert_path, key_path)) # 자체 서명 인증서 사용
    except FileNotFoundError: # 인증서 파일이 없을 경우 처리
        print(f"\n\nError starting Flask server: SSL certificate or key file not found.")
        print(f"Expected cert at: {cert_path}")
        print(f"Expected key at: {key_path}")
        print("Falling back to adhoc SSL context for development purposes...")
        try:
            app.run(host='0.0.0.0', port=443, debug=True, ssl_context='adhoc') # Fallback
        except Exception as adhoc_e:
             print(f"\n\nError starting Flask server even with adhoc SSL: {adhoc_e}\n\n")

    except Exception as e:
        print(f"\n\nError starting Flask server: {e}\n\n")