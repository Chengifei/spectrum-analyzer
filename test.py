import spectrum_analyzer as sa
import sys
import re

def error(msg):
    print(f"ERROR: {msg}")

def cast_to_time(s):
    s = s.strip()
    match = re.match(r'(?:([0-9]+)s)?\s*(?:([0-9]+)ms)?', s)
    if not match[1] and not match[2]:
        # interpret as a number
        try:
            return (0, int(s))
        except ValueError:
            raise ValueError(f"Wrong syntax in jump, got {s}")
    else:
        return (int(match[1]) if match[1] is not None else 0,
                int(match[2]) if match[2] is not None else 0)

try:
    f = sa.wav_file(sys.argv[1])
except IndexError:
    raise ValueError("Start the script with the file name")

ar = None
ar_tlen = 0
while True:
    In = input('>>> ')
    In = In.split(' ', 1)
    try:
        if In[0] == 'j':
            try:
                f.jump_to(*cast_to_time(In[1]))
            except IndexError:
                error("jump takes one parameter <time>")
        elif In[0] == 'g':
            ar_tlen = int(In[1])
            ar = f.get(ar_tlen)
        elif In[0] == 'f':
            ar = sa.fft(ar, 4200 // (1000 // ar_tlen))
        elif In[0] == 'p':
            bmp = sa.bmp_file(len(ar), 100)
            bmp.array(ar)
            bmp.save("a.bmp")
        elif In[0] == 't':
            print(f"{f.tell()}ms")
        elif In[0] == '+':
            f.inc(int(In[1]))
        elif In[0] == 'i':
            print("File", sys.argv[1])
            print("Sample rate:", f.sample_rate)
            print("Channels:", f.channels)
        else:
            error(f"Unknown command {In[0]}")
    except Exception:
        print(sys.exc_info())
