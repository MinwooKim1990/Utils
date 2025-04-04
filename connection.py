# %%
from flask import Flask, request, send_file, abort, render_template, redirect, url_for
import os
import functools

app = Flask(__name__)

API_KEY = 'MinwooKim1990'
ALLOWED_DRIVES = ['D:/', 'E:/']
HIDDEN_NAMES = {'system volume information', '$recycle.bin'}

# 캐시: 디렉토리 트리 정보를 30초간 캐시
tree_cache = {}

def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        try:
            received_key = request.args.get('api_key', '')
            print(f"Received API key: '{received_key}', Length: {len(received_key)}")
            print(f"Expected API key: '{API_KEY}', Length: {len(API_KEY)}")
            
            # 디버깅용 문자 비교
            if len(received_key) == len(API_KEY):
                for i in range(len(received_key)):
                    if received_key[i] != API_KEY[i]:
                        print(f"Mismatch at position {i}: '{received_key[i]}' != '{API_KEY[i]}'")
            
            if received_key != API_KEY:
                print("API key mismatch")
                abort(401)
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Error in API key validation: {e}")
            abort(500)
    return decorated


def get_base_dir():
    drive = request.args.get('drive', 'E:/')
    if drive not in ALLOWED_DRIVES:
        drive = 'E:/'
    return drive

# build_tree: 한 단계 자식 폴더까지 읽어오기 (추가 확장이 필요하면 재귀호출 가능)
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
                                        'children': []  # 2단계까지만 표시
                                    })
                            node['children'] = children
                    except Exception:
                        pass
                tree.append(node)
    except Exception:
        pass
    return tree

def get_entries(drive, rel_path):
    full_path = os.path.join(drive, rel_path)
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
                entries.append({
                    'name': entry.name,
                    'is_dir': entry.is_dir(),
                    'rel_path': os.path.relpath(os.path.join(full_path, entry.name), drive).replace("\\", "/"),
                    'is_image': ext in image_extensions and not entry.is_dir()
                })
    except Exception:
        pass
    # 정렬 (기본은 이름순)
    sort_by = request.args.get('sort', 'name')
    if sort_by == 'name':
        entries.sort(key=lambda x: x['name'].lower())
    elif sort_by == 'date':
        entries.sort(key=lambda x: os.path.getmtime(os.path.join(drive, x['rel_path'])), reverse=True)
    elif sort_by == 'size':
        entries.sort(key=lambda x: os.path.getsize(os.path.join(drive, x['rel_path'])) if os.path.isfile(os.path.join(drive, x['rel_path'])) else 0, reverse=True)
    return entries

@app.route('/')
@require_api_key
def file_browser():
    drive = get_base_dir()
    rel_path = request.args.get('path', '')
    tree_data = build_tree(drive)
    tree_html = render_tree(tree_data, drive)
    entries = get_entries(drive, rel_path)
    return render_template('template.html',
                           entries=entries,
                           api_key=API_KEY,
                           current_path=rel_path,
                           tree_html=tree_html,
                           drive=drive,
                           allowed_drives=ALLOWED_DRIVES,
                           sort_by=request.args.get('sort', 'name'))

# 새 엔드포인트: 파일 리스트 영역만 반환 (partial template)
@app.route('/filelist')
@require_api_key
def file_list():
    drive = get_base_dir()
    rel_path = request.args.get('path', '')
    entries = get_entries(drive, rel_path)
    return render_template('file_list.html',
                           entries=entries,
                           api_key=API_KEY,
                           current_path=rel_path,
                           drive=drive,
                           sort_by=request.args.get('sort', 'name'))

# render_tree: 왼쪽 트리 HTML 생성 (기본적으로 하위 ul은 닫힌 상태)
def render_tree(tree, drive):
    html = '<ul>'
    for node in tree:
        if node['is_dir']:
            if node.get('children'):
                html += (
                    f'<li>'
                    f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
                    f'<a href="/?drive={drive}&path={node["rel_path"]}&api_key={API_KEY}" class="folder-link" data-path="{node["rel_path"]}">{node["name"]}</a>'
                    f'<ul style="display: none;">'
                )
                for child in node['children']:
                    if child['is_dir']:
                        html += (
                            f'<li>'
                            f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
                            f'<a href="/?drive={drive}&path={node["rel_path"]+"/"+child["name"]}&api_key={API_KEY}" class="folder-link" data-path="{node["rel_path"]+"/"+child["name"]}">{child["name"]}</a>'
                            f'<ul style="display: none;"></ul>'
                            f'</li>'
                        )
                    else:
                        html += (
                            f'<li>'
                            f'<a href="/?drive={drive}&path={node["rel_path"]+"/"+child["name"]}&api_key={API_KEY}" class="file-link">{child["name"]}</a>'
                            f'</li>'
                        )
                html += '</ul></li>'
            else:
                html += (
                    f'<li>'
                    f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
                    f'<a href="/?drive={drive}&path={node["rel_path"]}&api_key={API_KEY}" class="folder-link" data-path="{node["rel_path"]}">{node["name"]}</a>'
                    f'<ul style="display: none;"></ul>'
                    f'</li>'
                )
        else:
            html += f'<li><a href="/?drive={drive}&path={node["rel_path"]}&api_key={API_KEY}" class="file-link">{node["name"]}</a></li>'
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
                # 필터링 (숨김, 시스템, 임시 파일 등)
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
    
    # 자식 폴더를 li 태그로 구성한 HTML 반환 (필요에 따라 디자인 수정)
    html = ""
    for child in children:
        html += (
            f'<li>'
            f'<span class="toggle-arrow" onclick="toggleChildren(this)">▶️</span> '
            f'<a href="/?drive={drive}&path={child["rel_path"]}&api_key={API_KEY}" class="folder-link" data-path="{child["rel_path"]}">{child["name"]}</a>'
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
    return redirect(url_for('file_browser', drive=drive, path=rel_path, api_key=API_KEY))

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
        return redirect(url_for('file_browser', drive=drive, path=parent_path, api_key=API_KEY))
    else:
        abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# %%
