import os
import sys

url = sys.argv[1]
dest = sys.argv[2]

if not os.path.exists(dest):
    os.mkdir(dest)

id = url.split('=')[-1]

# check if the destination files already exist
if os.path.exists('%s/yt-%s_0001.png' % (dest, id)):
    print('skipping %s' % id)
    sys.exit(0)

# run yt-dlp on the url
rv = os.system('./yt-dlp -f 243 -o "yt-%s.webm" %s' % (id, url))
if rv != 0:
    print('yt-dlp failed')
    sys.exit(1)

# run ffmpeg on the downloaded file
rv = os.system('ffmpeg -i yt-%s.webm -vf scale=32:18:flags=fast_bilinear -r 0.25 %s/yt-%s_%%04d.png' % (id, dest, id))
if rv != 0:
    print('ffmpeg failed')
    sys.exit(1)

# remove the intermediate file
os.remove('yt-%s.webm' % id)
