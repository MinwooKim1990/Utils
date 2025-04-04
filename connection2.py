from flask import Flask, request, send_file, abort, render_template, redirect, url_for, session, Response, stream_with_context
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
# Correct imports based on API expectations and common usage
# from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch

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

# --- Gemini Chat Endpoint --- START
@app.route('/chat', methods=['POST'])
@require_api_key # Ensure the user is authenticated
def handle_chat():
    api_key = app.config.get('GEMINI_API_KEY')
    if not api_key:
        return {"error": "Gemini API 키가 설정되지 않았습니다."}, 500

    # Configure the Gemini client
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Gemini 구성 오류: {e}")
        return {"error": "Gemini 클라이언트 설정에 실패했습니다."}, 500

    # --- Handle request based on Content-Type --- START
    history = []
    user_message = None
    model_name = None
    image_part = None

    content_type = request.content_type

    if content_type.startswith('application/json'):
        data = request.json
        if data is None:
            return {"error": "잘못된 요청 형식입니다. Content-Type은 application/json 이어야 합니다."}, 400
        user_message = data.get('message')
        model_name = data.get('model')
        history = data.get('history', [])

    elif content_type.startswith('multipart/form-data'):
        user_message = request.form.get('message')
        model_name = request.form.get('model')
        history_str = request.form.get('history', '[]')
        try:
            history = json.loads(history_str)
            if not isinstance(history, list):
                 raise ValueError("History is not a list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing history from FormData: {e}")
            return {"error": f"잘못된 히스토리 형식입니다: {e}"}, 400

        if 'image' in request.files:
            image_file = request.files['image']
            try:
                # Validate MIME type briefly
                if not image_file.mimetype.startswith('image/'):
                    return {"error": "첨부된 파일이 이미지가 아닙니다."}, 400

                # Read image data
                image_bytes = image_file.read()
                img = Image.open(io.BytesIO(image_bytes))

                # Prepare image part for Gemini API
                image_part = {
                    "mime_type": image_file.mimetype,
                    "data": image_bytes
                }
                print(f"[Debug] Prepared image_part: mime_type={image_part['mime_type']}, data_length={len(image_part['data'])}")
                # Force using a vision-capable model when image is present
                # Note: Replace with the actual latest vision model if different
                print(f"Image detected. Forcing model to gemini-1.5-flash-latest (original: {model_name})")
                model_name = "gemini-1.5-flash-latest" # Or gemini-pro-vision

            except Exception as e:
                print(f"Error processing uploaded image: {e}")
                return {"error": f"이미지 처리 중 오류 발생: {e}"}, 500
        else:
            # FormData request but no image file found
            return {"error": "이미지 파일이 요청에 포함되지 않았습니다."}, 400
    else:
         return {"error": f"지원되지 않는 Content-Type: {content_type}"}, 415
    # --- Handle request based on Content-Type --- END

    # Validate history format (basic check)
    if not isinstance(history, list):
         return {"error": "잘못된 히스토리 형식입니다."}, 400
    # Add more validation if needed (e.g., check structure of each item)

    if user_message is None or model_name is None: # Check for None after extraction
        return {"error": "메시지 또는 모델 이름이 누락되었습니다."}, 400

    try:
        model = genai.GenerativeModel(model_name)

        # --- Prepare content for API call --- START
        # Combine history and the new prompt
        # Ensure history items have the correct format {role: ..., parts: [...]} expected by API
        # Our current JS history format matches this.
        prompt_parts = []
        # Ensure user_message is added as a dict part if it exists
        if user_message:
            prompt_parts.append({"text": user_message})
        # Append image part AFTER text part, if it exists
        # Gemini API prefers text before image in the parts list for multimodal input
        if image_part: 
            prompt_parts.append(image_part) 

        if not prompt_parts:
             return {"error": "메시지와 이미지가 모두 비어있습니다."}, 400

        print(f"[Debug] Current prompt_parts (before adding to history list): {prompt_parts}") # Renamed log message

        # Construct the full conversation payload for generate_content
        validated_history = []
        for entry in history:
            role = entry.get('role')
            parts = entry.get('parts')
            if role in ['user', 'model'] and isinstance(parts, list) and parts:
                validated_history.append({'role': role, 'parts': parts})
            else:
                print(f"[Warning] Skipping invalid history entry: {entry}")

        contents_payload = validated_history + [{'role': 'user', 'parts': prompt_parts}]
         
        print(f"Sending to Gemini ({model_name}). History length: {len(validated_history)}, Image attached: {image_part is not None}")
        print(f"[Debug] Contents Payload being sent to API:", contents_payload)

        # Use generate_content which handles history and multimodality
        # Remove the config parameter
        response = model.generate_content(
            contents_payload
        )

        # --- Streaming Response --- START
        def stream_response(api_response):
            try:
                for chunk in api_response:
                    if chunk.text:
                        # Send data using Server-Sent Events (SSE) format
                        yield f"data: {json.dumps({'response': chunk.text})}\n\n"
            except Exception as e:
                print(f"Error during streaming: {e}")
                # Send an error event (optional)
                yield f"data: {json.dumps({'error': f'스트리밍 중 오류 발생: {e}'})}\n\n"
            finally:
                 # Signal end of stream (optional, client can handle stream end)
                 yield f"event: end\ndata: Stream ended\n\n"

        # Check if response is streamable (depends on API/model behavior)
        # For generate_content with stream=True, the response itself is the iterator
        try:
             # Call generate_content with stream=True
             stream = model.generate_content(
                 contents_payload,
                 stream=True
             )
             # Return a streaming response
             return Response(stream_with_context(stream_response(stream)), mimetype='text/event-stream')

        except Exception as e:
             print(f"Gemini API 스트리밍 호출 오류 ({model_name}): {e}")
             # Return a non-streaming error if the initial call fails
             error_detail = str(e)
             if "API key not valid" in error_detail:
                  error_msg = "Gemini API 키가 유효하지 않습니다. .env 파일을 확인하세요."
             elif "quota" in error_detail.lower():
                  error_msg = "Gemini API 할당량이 초과되었습니다."
             else:
                  error_msg = f"Gemini API ({model_name}) 스트리밍 호출 중 오류 발생: {error_detail}"
             # Send error as a single JSON response for non-streaming failures
             # Client-side fetch needs to handle this potentially non-streaming error
             return {"error": error_msg}, 500
        # --- Streaming Response --- END

    except Exception as e:
        # This outer try-except might catch errors before streaming starts
        print(f"Gemini API 호출 준비 오류 ({model_name}): {e}")
        error_detail = str(e)
        error_msg = f"Gemini API ({model_name}) 호출 준비 중 오류 발생: {error_detail}"
        return {"error": error_msg}, 500
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)