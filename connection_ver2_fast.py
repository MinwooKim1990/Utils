from connection_ver2 import app as base_app, get_validated_path
import os, mimetypes
from flask import request, abort, Response, stream_with_context, redirect, url_for, session

app = base_app

# 기존 엔드포인트 제거 (중복 방지)
app.view_functions.pop('stream_file', None)
app.view_functions.pop('upload_file', None)

# ----------- 다운로드 최적화 (4MB 청크) -----------
@app.route('/stream/<path:rel_path>')
def stream_file(rel_path):
    drive = request.args.get('drive', 'E:/')
    if drive not in ['D:/', 'E:/', 'I:/']:
        drive = 'E:/'
    full_path = get_validated_path(drive, rel_path)
    if not os.path.isfile(full_path):
        abort(404)
    range_header = request.headers.get('Range', None)
    size = os.path.getsize(full_path)
    chunk_size = 4 * 1024 * 1024
    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type:
        mime_type = 'application/octet-stream'
    start = 0
    end = size - 1
    status = 200
    if range_header:
        try:
            range_val = range_header.replace('bytes=', '')
            parts = range_val.split('-')
            start = int(parts[0])
            if len(parts) > 1 and parts[1]:
                end = int(parts[1])
            else:
                end = size - 1
            if start >= size or end >= size or start > end:
                raise ValueError
            status = 206
        except:
            abort(416)
    length = end - start + 1
    def generate():
        with open(full_path, 'rb') as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                yield data
                remaining -= len(data)
    resp = Response(stream_with_context(generate()), status, mimetype=mime_type)
    resp.headers.add('Content-Length', str(length))
    resp.headers.add('Accept-Ranges', 'bytes')
    if status == 206:
        resp.headers.add('Content-Range', f'bytes {start}-{end}/{size}')
    return resp

# ----------- 업로드 최적화 (청크 저장) -----------
@app.route('/upload', methods=['POST'])
def upload_file():
    drive = request.args.get('drive', 'E:/')
    if drive not in ['D:/', 'E:/', 'I:/']:
        drive = 'E:/'
    rel_path = request.args.get('path', '')
    save_dir = os.path.join(drive, rel_path)
    os.makedirs(save_dir, exist_ok=True)
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename
        if not filename:
            abort(400)
        filename_parts = filename.rsplit('.', 1)
        if len(filename_parts) == 2 and filename_parts[0]:
            new_name = f"{filename_parts[0]}_MAC.{filename_parts[1]}"
        elif len(filename_parts) == 1:
            new_name = f"{filename}_MAC"
        else:
            new_name = f"_MAC.{filename_parts[1]}"
        save_path = os.path.join(save_dir, new_name)
        try:
            with open(save_path, 'wb') as f:
                chunk_size = 4 * 1024 * 1024
                while True:
                    chunk = file.stream.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            print(f"업로드 저장 실패: {e}")
            abort(500)
    else:
        abort(400)
    from stringencode import encode_string
    encrypted_key = encode_string('MinwooKim1990', session.get('encrypt_password', ''))
    return redirect(url_for('file_browser', drive=drive, path=rel_path, api_key=encrypted_key))

# ----------- Gunicorn + gevent 권장 -----------
# gunicorn -w 4 -k gevent -b 0.0.0.0:443 --certfile=cert.pem --keyfile=key.pem Utils.connection3:app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=443, debug=True, ssl_context='adhoc')