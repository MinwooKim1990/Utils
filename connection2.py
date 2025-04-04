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

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Flask 세션 사용을 위한 secret key

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
        
        # Safari 브라우저와 디바이스 파라미터 확인
        is_safari = request.args.get('is_safari')
        device = request.args.get('device')
        
        # 세션에 이미 인증됨을 저장하고 있는지 확인
        if 'authenticated' in session and session['authenticated']:
            return f(*args, **kwargs)
            
        # Safari 브라우저와 디바이스 파라미터 검증
        if is_safari and device:
            # 해당 파라미터를 URL에서 숨기기 위해 세션에 저장
            session['is_safari'] = is_safari
            session['device'] = device
            
            # 필요한 값과 일치하는지 확인
            if is_safari != REQUIRED_BROWSER or device != REQUIRED_DEVICE:
                return render_template('login.html', error="인증되지 않은 기기입니다. 접근이 거부되었습니다.")
        
        # 세션에 암호가 없고 plain API_KEY로 접속한 경우
        if 'encrypt_password' not in session:
            if received_key == API_KEY:
                # Safari 브라우저와 디바이스 파라미터가 올바른 경우에만 진행
                if session.get('is_safari') == REQUIRED_BROWSER and session.get('device') == REQUIRED_DEVICE:
                    random_password = secrets.token_hex(8)  # 랜덤 암호 생성
                    session['encrypt_password'] = random_password
                    session['authenticated'] = True  # 인증 성공 기록
                    encrypted_api_key = encode_string(API_KEY, random_password)
                    # 동일한 URL로 암호화된 API 키를 포함하여 리다이렉트
                    args_dict = request.args.to_dict()
                    args_dict['api_key'] = encrypted_api_key
                    # Safari와 디바이스 파라미터는 URL에서 제거
                    if 'is_safari' in args_dict:
                        del args_dict['is_safari']
                    if 'device' in args_dict:
                        del args_dict['device']
                    return redirect(url_for(request.endpoint, **args_dict))
                else:
                    return render_template('login.html', error="Safari 브라우저 인증이 필요합니다.")
            else:
                # API 키가 제공되었지만 맞지 않는 경우
                if received_key:
                    return render_template('login.html', error="잘못된 API 키입니다.")
                # API 키가 제공되지 않은 경우
                return render_template('login.html')
        else:
            random_password = session['encrypt_password']
            encrypted_api_key = encode_string(API_KEY, random_password)
            # 만약 여전히 plain API_KEY로 접속 시, 암호화된 키로 리다이렉트
            if received_key == API_KEY:
                if session.get('is_safari') == REQUIRED_BROWSER and session.get('device') == REQUIRED_DEVICE:
                    session['authenticated'] = True  # 인증 성공 기록
                    args_dict = request.args.to_dict()
                    args_dict['api_key'] = encrypted_api_key
                    # Safari와 디바이스 파라미터는 URL에서 제거
                    if 'is_safari' in args_dict:
                        del args_dict['is_safari']
                    if 'device' in args_dict:
                        del args_dict['device']
                    return redirect(url_for(request.endpoint, **args_dict))
                else:
                    return render_template('login.html', error="Safari 브라우저 인증이 필요합니다.")
            elif received_key == encrypted_api_key:
                if session.get('is_safari') == REQUIRED_BROWSER and session.get('device') == REQUIRED_DEVICE:
                    session['authenticated'] = True  # 인증 성공 기록
                    return f(*args, **kwargs)
                else:
                    return render_template('login.html', error="Safari 브라우저 인증이 필요합니다.")
            else:
                return render_template('login.html', error="인증에 실패했습니다.")
    return decorated

@app.route('/')
@require_api_key
def file_browser():
    drive = get_base_dir()
    rel_path = request.args.get('path', '')
    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
    tree_data = build_tree(drive)
    tree_html = render_tree(tree_data, drive, encrypted_key)
    entries = get_entries(drive, rel_path)
    return render_template('template.html',
                           entries=entries,
                           api_key=encrypted_key,
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
    rel_path = request.args.get('path')
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
    filename = file.filename.rsplit('.', 1)
    new_name = f"{filename[0]}_MAC.{filename[1]}" if len(filename) == 2 else f"{file.filename}_MAC"
    save_path = os.path.join(full_path, new_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    encrypted_key = encode_string(API_KEY, session['encrypt_password'])
    return redirect(url_for('file_browser', drive=drive, path=rel_path, api_key=encrypted_key))

@app.route('/delete')
@require_api_key
def delete_file():
    drive = request.args.get('drive', 'E:/')
    if drive not in ALLOWED_DRIVES:
        drive = 'E:/'
    base_dir = drive
    rel_path = request.args.get('path')
    full_path = os.path.join(base_dir, rel_path)
    if os.path.isfile(full_path):
        os.remove(full_path)
        parent_path = os.path.dirname(rel_path)
        encrypted_key = encode_string(API_KEY, session['encrypt_password'])
        return redirect(url_for('file_browser', drive=drive, path=parent_path, api_key=encrypted_key))
    else:
        abort(404)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)