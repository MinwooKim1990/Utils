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
# Correct imports based on API expectations and common usage
# from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch

# <<< RAG Imports >>>
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
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
ALLOWED_DRIVES = ['D:/', 'E:/']
HIDDEN_NAMES = {'system volume information', '$recycle.bin'}

# 인증된 디바이스 설정
REQUIRED_DEVICE = '6030'
REQUIRED_BROWSER = 'true'

tree_cache = {}

# <<< RAG Initialization >>>
# Load Sentence Transformer model (adjust model name if needed)
# Using a multilingual model suitable for Korean and English
try:
    print("Loading Sentence Transformer model...")
    # Consider smaller/faster models if resource usage is a concern
    # e.g., 'distiluse-base-multilingual-cased-v1' or specific Ko models
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    print(f"FATAL: Failed to load Sentence Transformer model: {e}")
    # Optionally exit or disable RAG features if model loading fails
    embedding_model = None

# In-memory store for document chunks and embeddings
# Structure: { instanceId: { 'chunks': [chunk1, chunk2, ...], 'embeddings': [emb1, emb2, ...] } }
document_vector_stores = {}

def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
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

    # Consider using get_validated_path here for security and existence check
    # For now, just fix the type error by ensuring rel_path is a str
    full_path = os.path.join(base_dir, rel_path)
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
    base_dir = drive
    rel_path = request.args.get('path', '')
    full_path = os.path.join(base_dir, rel_path)
    if 'file' not in request.files:
        abort(400)
    file = request.files['file']
    # Check if filename exists and is not None
    if not file or not file.filename:
        abort(400, description="Invalid file or filename.")

    filename_parts = file.filename.rsplit('.', 1)
    # Handle case with no extension or only extension
    if len(filename_parts) == 2 and filename_parts[0]:
        new_name = f"{filename_parts[0]}_MAC.{filename_parts[1]}"
    elif len(filename_parts) == 1: # No extension
        new_name = f"{file.filename}_MAC"
    else: # Edge case, e.g., ".gitignore" -> parts = ['', 'gitignore']
         new_name = f"_MAC.{filename_parts[1]}" # Or handle differently

    save_path = os.path.join(full_path, new_name)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        abort(500, description="Failed to save uploaded file.")

    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
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

    # Consider using get_validated_path here for security and existence check
    full_path = os.path.join(base_dir, rel_path)

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
    drive = get_base_dir()
    full_path = get_validated_path(drive, rel_path) # Use validated path

    if not os.path.isfile(full_path):
        abort(404, "Requested path is not a file.")

    range_header = request.headers.get('Range', None)
    size = os.path.getsize(full_path)
    chunk_size = 1024 * 1024 # 1MB chunks (adjust as needed)

    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type:
        mime_type = 'application/octet-stream' # Default if type can't be guessed

    start_byte = 0
    end_byte = size - 1 # Default to entire file if no range or invalid range
    status_code = 200 # OK by default

    if range_header:
        try:
            # Simple parsing for "bytes=start-[end]" or "bytes=start-"
            range_str = range_header.replace('bytes=', '')
            parts = range_str.split('-')
            start_byte = int(parts[0])
            if len(parts) > 1 and parts[1]: # Check if end byte is specified
                end_byte = int(parts[1])
            else: # No end byte specified, stream to the end (but browser usually requests smaller ranges)
                end_byte = size - 1 # For range header, let's respect it up to the end

            # Validate range
            if start_byte >= size or end_byte >= size or start_byte > end_byte:
                 raise ValueError("Invalid range")
            status_code = 206 # Partial Content

        except ValueError:
            print(f"Invalid Range header received: {range_header}, sending 416")
            abort(416, description="Requested Range Not Satisfiable")

    length = end_byte - start_byte + 1

    def generate_chunks():
        with open(full_path, 'rb') as f:
            f.seek(start_byte)
            bytes_to_read = length
            while bytes_to_read > 0:
                # Use smaller read size for smoother streaming potentially
                read_size = min(bytes_to_read, chunk_size) 
                chunk = f.read(read_size)
                if not chunk:
                    break
                yield chunk
                bytes_to_read -= len(chunk)

    resp = Response(stream_with_context(generate_chunks()), status_code, mimetype=mime_type)
    resp.headers.add('Content-Length', str(length))
    resp.headers.add('Accept-Ranges', 'bytes')
    if status_code == 206:
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
        # Try common encodings, prioritizing UTF-8 and system default (like cp949 on Korean Windows)
        encodings_to_try = [encoding, 'utf-8', 'cp949', 'euc-kr', 'latin-1']
        encodings_to_try = [enc for enc in encodings_to_try if enc]
        encodings_to_try = list(dict.fromkeys(encodings_to_try))

        content = None
        detected_encoding = None
        # Limit the amount read for preview for very large text files?
        # max_preview_bytes = 1 * 1024 * 1024 # e.g., 1MB limit for text preview

        for enc in encodings_to_try:
            try:
                print(f"Attempting to read {os.path.basename(full_path)} with encoding: {enc}")
                with open(full_path, 'r', encoding=enc) as f:
                    # content = f.read(max_preview_bytes) # Read limited amount
                    content = f.read() # Read full content for now
                detected_encoding = enc
                print(f"Successfully read with {enc}")
                break
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError with {enc}")
                continue
            except LookupError:
                 print(f"Codec not found: {enc}")
                 continue
            except Exception as e:
                 print(f"Error reading file {full_path} with encoding {enc}: {e}")
                 continue

        if content is None:
            try:
                print(f"Trying final read as binary with utf-8 replacement")
                with open(full_path, 'rb') as f:
                     # binary_content = f.read(max_preview_bytes) # Limit read size
                     binary_content = f.read()
                content = binary_content.decode('utf-8', errors='replace')
                detected_encoding = 'utf-8 (with replacements)'
                print("Read as binary succeeded.")
            except Exception as e:
                 print(f"Final attempt to read file {full_path} as binary failed: {e}")
                 return Response(f"Error reading file: Could not decode content.", status=500, mimetype='text/plain')

        # Use iframe compatible HTML for preview
        # We are embedding this in an iframe later, so a full HTML doc is fine.
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
                 /* Optional: Add basic dark mode support inside the iframe based on parent */
                 /* This requires JS in the iframe or passing a theme param */
            </style>
        </head>
        <body>
            <pre>{escape(content)}</pre>
        </body>
        </html>
        """
        return Response(html_content, mimetype='text/html') # Send as HTML

    except Exception as e:
        print(f"Error previewing text file {full_path}: {e}")
        error_message = f"Error generating file preview for {escape(os.path.basename(full_path))}: {escape(str(e))}"
        # Return error as plain text
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
    encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                text = f.read()
            print(f"Extracted {len(text)} characters from plain text file: {os.path.basename(file_path)} using {enc}")
            return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error extracting text from plain file {file_path} with {enc}: {e}")
            # Stop trying if a non-decode error occurs
            return None
    # If all encodings fail, try reading as binary with replacement
    try:
        with open(file_path, 'rb') as f:
            binary_content = f.read()
        text = binary_content.decode('utf-8', errors='replace')
        print(f"Extracted {len(text)} characters from plain text file (binary fallback): {os.path.basename(file_path)}")
        return text
    except Exception as e:
        print(f"Final fallback error reading plain text file {file_path}: {e}")
        return None

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks with overlap (simple character-based)."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap # Move start forward with overlap
        if start >= len(text): # Prevent infinite loop on very short overlaps
             break
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

def get_embeddings(texts):
    """Generates embeddings for a list of texts using the loaded model."""
    if not embedding_model or not texts:
        return None
    try:
        embeddings = embedding_model.encode(texts, convert_to_tensor=False) # Get numpy arrays
        print(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Finds top_k most relevant chunks based on cosine similarity."""
    if query_embedding is None or chunk_embeddings is None or not chunks:
        return []
    # Calculate cosine similarities
    # Normalize embeddings for efficient cosine similarity calculation
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1)[:, np.newaxis]
    similarities = np.dot(chunk_norms, query_norm)

    # Get top_k indices
    # Use partition for efficiency if top_k is small relative to total chunks
    # For simplicity, using argsort here
    sorted_indices = np.argsort(similarities)[::-1] # Sort descending
    top_indices = sorted_indices[:top_k]

    relevant_chunks = [(chunks[i], similarities[i]) for i in top_indices]
    print(f"Found {len(relevant_chunks)} relevant chunks with similarities: {[f'{s:.3f}' for _, s in relevant_chunks]}")
    return [chunk for chunk, _ in relevant_chunks] # Return only the chunk text

# --- RAG Helper Functions --- END

# --- Document Processing Endpoint --- START
@app.route('/process_document', methods=['POST'])
@require_api_key
def process_document():
    if not embedding_model:
        return {"error": "Embedding model not loaded."}, 500

    data = request.json
    rel_path = data.get('relPath')
    instance_id = data.get('instanceId')
    file_name = data.get('fileName') # Get filename for extension check

    if not rel_path or not instance_id or not file_name:
        return {"error": "Missing required parameters (relPath, instanceId, fileName)."}, 400

    drive = get_base_dir() # Get the current drive
    full_path = get_validated_path(drive, rel_path) # Validate and get full path
    file_ext = os.path.splitext(file_name)[1].lower()

    print(f"Processing document: {rel_path} for instance: {instance_id}")

    text = None
    if file_ext == '.pdf':
        text = extract_text_from_pdf(full_path)
    elif file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.log', '.sh', '.bat', '.ps1']: # Add more supported text extensions here
        text = extract_text_from_plain(full_path)
    else:
        # Note: Add support for docx, pptx etc. here if needed later
        print(f"Skipping processing for unsupported file type: {file_ext}")
        return {"error": f"Unsupported file type for RAG: {file_ext}"}, 415

    if text is None:
        print(f"Failed to extract text from {rel_path}")
        return {"error": "Failed to extract text from the document."}, 500

    chunks = chunk_text(text) # Use default chunk size/overlap for now
    if not chunks:
        print(f"No chunks generated for {rel_path}")
        return {"error": "Document is empty or could not be chunked."}, 400

    embeddings = get_embeddings(chunks)
    if embeddings is None:
        print(f"Failed to generate embeddings for {rel_path}")
        return {"error": "Failed to generate embeddings for the document."}, 500

    # Store chunks and embeddings in memory
    document_vector_stores[instance_id] = {
        'chunks': chunks,
        'embeddings': embeddings,
        'rel_path': rel_path # Store path for potential future use/validation
    }

    print(f"Successfully processed and stored document for instance {instance_id}. Chunks: {len(chunks)}, Embeddings shape: {embeddings.shape}")
    return {"message": f"문서 처리 완료: {len(chunks)}개 청크 생성됨"}, 200

# --- Document Processing Endpoint --- END

# --- Gemini Chat Endpoint --- START
@app.route('/chat', methods=['POST'])
@require_api_key
def chat():
    if not app.config.get('GEMINI_API_KEY'):
        return jsonify({"error": "API key not configured."}), 500

    # Determine content type and parse data
    content_type = request.content_type
    data = None
    image_file = None
    instance_id = None # <<< Added for RAG context

    if content_type.startswith('application/json'):
        data = request.json
        instance_id = data.get('instanceId') # <<< Get instanceId from JSON
    elif content_type.startswith('multipart/form-data'):
        data = request.form.to_dict()
        image_file = request.files.get('image')
        instance_id = data.get('instanceId') # <<< Get instanceId from FormData
        # Need to parse history from string if it's FormData
        if 'history' in data:
            try:
                data['history'] = json.loads(data['history'])
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid history format in form data."}), 400
    else:
        return jsonify({"error": "Unsupported Content-Type"}), 415

    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    message = data.get('message', '')
    model_name = data.get('model', 'gemini-1.5-flash-latest') # Default to flash
    history = data.get('history', []) # Expecting list of {'role': 'user'/'model', 'parts': [{'text': '...'}]}

    # Validate history format (simple check)
    if not isinstance(history, list):
        history = [] # Reset if format is wrong

    print(f"Received chat request. Model: {model_name}, InstanceID: {instance_id}, History Length: {len(history)}, Image Attached: {image_file is not None}")

    # <<< RAG Logic Integration >>>
    rag_context = ""
    is_document_chat = False
    if instance_id and instance_id in document_vector_stores:
        is_document_chat = True
        print(f"Instance {instance_id} found in document store. Applying RAG.")
        doc_store = document_vector_stores[instance_id]
        if not message:
            print("Skipping RAG for empty message in document chat.")
        elif not embedding_model:
             print("Skipping RAG because embedding model is not loaded.")
        else:
            # Generate embedding for the user's query
            query_embedding = get_embeddings([message])
            if query_embedding is not None:
                relevant_chunks = find_relevant_chunks(
                    query_embedding[0], # Get the single query embedding
                    doc_store['embeddings'],
                    doc_store['chunks'],
                    top_k=3 # Retrieve top 3 relevant chunks
                )
                if relevant_chunks:
                    rag_context = "\n\n-- 문서 내용 --\n"
                    rag_context += "\n\n".join(relevant_chunks)
                    rag_context += "\n--------------\n"
                    print(f"Added RAG context for instance {instance_id}")
                else:
                    print("No relevant chunks found for the query.")
            else:
                 print("Failed to generate query embedding.")

    # --- Model Initialization --- (Moved after RAG context generation)
    try:
        genai.configure(api_key=app.config['GEMINI_API_KEY'])
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return jsonify({"error": f"Failed to initialize model: {e}"}), 500

    # --- Prepare Prompt and History --- 
    # Construct the final prompt for the model
    # For RAG, prepend context to the latest user message
    # For image chats, handle image data
    # For regular chats, just use the message

    chat_session = model.start_chat(history=history)
    prompt_parts = []

    # <<< MODIFIED: Construct a single text prompt including RAG context >>>
    final_prompt_text = ""

    # Append RAG context if available
    if rag_context:
        final_prompt_text += rag_context # Prepend RAG context

    # Append user message text
    if message:
        final_prompt_text += message # Append user message after RAG context

    # Handle image input (if not a document chat and image exists)
    # Document chats currently do not support simultaneous image input in this flow
    image_prompt_part = None # Store potential image part separately
    if image_file and not is_document_chat:
        try:
            img_bytes = image_file.read()
            # Prepare image part for Gemini API directly
            image_prompt_part = {
                "mime_type": image_file.mimetype,
                "data": img_bytes
            }

            print(f"Processing image: {image_file.filename}, size: {len(img_bytes)} bytes")
            # Determine appropriate prompt based on history
            if not history or len(history) == 0: # Initial image upload
                # Revised initial prompt for more detailed analysis
                initial_prompt = (
                    "이 이미지를 자세히 분석하고 주요 요소, 객체, 장면, 분위기 등을 설명해주세요. "
                    "가능하다면 텍스트(OCR)도 추출해주세요. "
                    "분석이 끝나면 이 이미지에 대해 무엇이 궁금한지 물어봐주세요."
                )
                # Prepend the initial prompt to the main text
                final_prompt_text = initial_prompt + ("\n\n" + final_prompt_text if final_prompt_text else "")
                print("Using detailed initial analysis prompt.")
            else:
                # Follow-up question with image context
                print("Sending follow-up message with image context.")
        except Exception as e:
            print(f"Error processing image: {e}")
            # Decide how to handle: return error or proceed without image?
            # For now, let's proceed without the image if processing fails
            image_prompt_part = None # Discard image part on error
            # Optionally add an error message to the prompt?
            # final_prompt_text += "\n(이미지 처리 실패)" # Add failure notice to text
            pass # Continue without image if processing fails

    # --- Construct final prompt parts for API call ---
    final_api_parts = []
    if final_prompt_text: # Add the combined text part if it's not empty
        final_api_parts.append(final_prompt_text)
    if image_prompt_part: # Add the image part if it exists
        final_api_parts.append(image_prompt_part)

    # --- Generate Response --- 
    if not final_api_parts:
         # Handle cases where there's nothing to send (e.g., empty message and no image/context)
         print("Prompt is empty, sending default response.")
         def empty_stream():
            yield f'event: data\ndata: {{"response": "메시지를 입력해주세요."}}\n\n'
            yield f'event: end\ndata: {{}}\n\n'
         return Response(empty_stream(), mimetype='text/event-stream')

    print(f"Sending to Gemini. Final API Parts Count: {len(final_api_parts)}")
    # print(f"Prompt Parts Content (text only): {[p for p in prompt_parts if isinstance(p, str)]}")

    # --- Streaming Response --- 
    def stream():
        try:
            # Use generate_content with stream=True
            response_stream = chat_session.send_message(final_api_parts, stream=True)

            for chunk in response_stream:
                if chunk.parts:
                    text_part = chunk.parts[0].text
                    # print(f"Stream chunk: {text_part[:50]}...") # Debug output
                    # Send data formatted as Server-Sent Events (SSE)
                    yield f'data: {json.dumps({"response": text_part})}\n\n' # <<< Use standard SSE format
                # Add safety rating check if needed
                # if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                #    print(f"Request blocked: {chunk.prompt_feedback.block_reason}")
                #    yield f'data: {json.dumps({"error": f"요청 차단됨: {chunk.prompt_feedback.block_reason}"})}\n\n'
                #    return # Stop streaming on block

            # Signal the end of the stream
            yield f'event: end\ndata: Stream ended\n\n' # <<< Signal end with standard event
            print("Stream finished.")

        except exceptions.GoogleAPICallError as e:
            # Handle specific API call errors (e.g., quota, unavailable)
            print(f"Google API Call Error: {e}")
            error_message = f"API 호출 오류: {e.message}"
            if hasattr(e, 'status_code') and e.status_code == 503: # Overloaded
                 error_message = "모델 사용량 초과(503). 잠시 후 다시 시도해주세요."
            yield f'data: {json.dumps({"error": error_message})}\n\n'
            yield f'event: end\ndata: Error occurred\n\n' # <<< Signal end
        except exceptions.BlockedPromptException as e:
            # Handle blocked prompts specifically
            print(f"Blocked Prompt Error: {e}")
            yield f'data: {json.dumps({"error": f"요청 차단됨 (프롬프트): {e}"})}\n\n'
            yield f'event: end\ndata: Error occurred\n\n' # <<< Signal end
        except Exception as e:
            # Catch-all for other unexpected errors during streaming
            print(f"Streaming Error: {e}")
            # traceback.print_exc() # Optional detailed traceback
            yield f'data: {json.dumps({"error": f"스트리밍 중 오류 발생: {e}"})}\n\n'
            yield f'event: end\ndata: Error occurred\n\n' # <<< Signal end

    return Response(stream(), mimetype='text/event-stream')

# --- Gemini Chat Endpoint --- END

# --- Document Cache Clearing Endpoint --- START
@app.route('/clear_document_cache', methods=['POST'])
@require_api_key
def clear_document_cache():
    data = request.json
    instance_id = data.get('instanceId')

    if not instance_id:
        return jsonify({"error": "Missing instanceId parameter."}), 400

    if instance_id in document_vector_stores:
        del document_vector_stores[instance_id]
        print(f"Cleared document cache for instance: {instance_id}")
        # Also trigger garbage collection potentially?
        # import gc
        # gc.collect()
        return jsonify({"message": f"Cache cleared for instance {instance_id}"}), 200
    else:
        print(f"Attempted to clear cache for non-existent instance: {instance_id}")
        return jsonify({"error": "Instance ID not found in cache."}), 404

# --- Document Cache Clearing Endpoint --- END

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)