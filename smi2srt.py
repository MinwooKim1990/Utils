#!/usr/bin/env python
import codecs
import re
import cchardet
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--remove_original",
                    help="remove original smi file when converting succeed", default=False, action="store_true")
parser.add_argument("-i", "--ignore", help="ignore decoding error",
                    default=False, action="store_true")
parser.add_argument(
    "dir", help="target directory. ex) ./smi2srt <OPTIONS> ./Movies", default="./")

args = parser.parse_args()

PATH = args.dir
REMOVE_OPTION = args.remove_original
DECODE_ERRORS = 'ignore' if args.ignore else 'strict'


def parse(smi):
    def get_languages():
        pattern = re.compile(r'<p class=(\w+)>', flags=re.I)
        langs = list(sorted(set(pattern.findall(smi))))
        return langs

    def remove_tag(matchobj):
        matchtag = matchobj.group().lower()
        keep_tags = ['font', 'b', 'i', 'u']
        for keep_tag in keep_tags:
            if keep_tag in matchtag:
                return matchtag
        return ''

    def parse_p(item):
        pattern = re.compile(r'<p class=(\w+)>(.+)', flags=re.I | re.DOTALL)
        parsed = {}
        for match in pattern.finditer(item):
            lang = match.group(1)
            content = match.group(2)
            content = content.replace('\r', '')
            content = content.replace('\n', '')
            content = re.sub('<br ?/?>', '\n', content, flags=re.I)
            content = re.sub('<[^>]+>', remove_tag, content)
            parsed[lang] = content
        return parsed

    data = []
    try:
        pattern = re.compile(
            r'<sync (start=\d+)\s?(end=\d+)?>', flags=re.I | re.S)
        start_end_content = pattern.split(smi)[1:]
        start = start_end_content[::3]
        end = start_end_content[1::3]
        content = start_end_content[2::3]
        for s, e, c in zip(start, end, content):
            datum = {}
            datum['start'] = int(s.split('=')[1])
            datum['end'] = int(e.split('=')[1]) if e is not None else None
            datum['content'] = parse_p(c)
            data.append(datum)
        return get_languages(), data
    except:
        print('Conversion ERROR: maybe this file is not supported.')

    return get_languages(), data


def convert(data, lang):  # written by ncianeo
    def ms_to_ts(time):
        time = int(time)
        ms = time % 1000
        s = int(time/1000) % 60
        m = int(time/1000/60) % 60
        h = int(time/1000/60/60)
        return (h, m, s, ms)

    srt = ''
    sub_nb = 1
    for i in range(len(data)-1):
        try:
            if i > 0:
                if data[i]['start'] < data[i-1]['start']:
                    continue
            if data[i]['content'][lang] != '&nbsp;':
                srt += str(sub_nb)+'\n'
                sub_nb += 1
                if data[i]['end'] is not None:
                    srt += '%02d:%02d:%02d,%03d' % ms_to_ts(
                        data[i]['start'])+' --> '+'%02d:%02d:%02d,%03d\n' % ms_to_ts(data[i]['end'])
                else:
                    if int(data[i+1]['start']) > int(data[i]['start']):
                        srt += '%02d:%02d:%02d,%03d' % ms_to_ts(
                            data[i]['start'])+' --> '+'%02d:%02d:%02d,%03d\n' % ms_to_ts(data[i+1]['start'])
                    else:
                        srt += '%02d:%02d:%02d,%03d' % ms_to_ts(
                            data[i]['start'])+' --> '+'%02d:%02d:%02d,%03d\n' % ms_to_ts(int(data[i]['start'])+1000)
                srt += data[i]['content'][lang]+'\n\n'
        except:
            continue
    return srt


print('media library path:', PATH)
success = []
fail = []
print('finding and converting started...')
for p, w, f in os.walk(PATH):
    for file_name in f:
        if file_name[-4:].lower() == '.smi':
            print('processing %s' % os.path.join(p, file_name))
            try:
                with open(os.path.join(p, file_name), 'rb') as smi_file:
                    smi_raw = smi_file.read()
                    encoding = cchardet.detect(smi_raw)
                smi = smi_raw.decode(
                    encoding['encoding'], errors=DECODE_ERRORS)
                langs, data = parse(smi)
                for lang in langs:
                    lang_code = lang.replace('CC', '')
                    SUFFIX = '.'+(lang_code.lower() if len(
                        lang_code) == 2 else lang_code[:2].lower() + '_' + lang_code[2:].upper())
                    srt_file = codecs.open(os.path.join(p, os.path.splitext(
                        file_name)[0]+SUFFIX+'.srt'), 'w', encoding='utf-8')
                    srt_file.write(convert(data, lang))
                success.append(file_name)
                if REMOVE_OPTION:
                    os.remove(os.path.join(p, file_name))
            except:
                fail.append(file_name)

smi_list = list(set(success) | set(fail))
print('\nfound .smi subtitles:')
for smi in smi_list:
    print(smi)

print('\nworked .smi subtitles:')
for smi in success:
    print(smi)

print('\nfailed .smi subtitles:')
for smi in fail:
    print(smi)

if REMOVE_OPTION:
    print('\nworked smi files are removed due to removal option')