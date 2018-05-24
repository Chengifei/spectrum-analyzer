#!/usr/local/bin/python3
import spectrum_analyzer as sa
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import sys
import re

def error(msg):
    print(f"ERROR: {msg}")

def cast_to_time(s):
    delim2 = s.find('ms') # match longest first
    delim1 = s.find('s', 0, delim2 if delim2 != -1 else None)
    if delim2 != -1:
        # ms present, check s
        if delim1 != -1:
            # both present
            return int(s[:delim1]) * 1000 + int(s[delim1 + 1:delim2])
        else:
            # only ms
            return int(s[:delim2])
    elif delim1 != -1:
        # ms not present, s present
        return int(s[:delim1]) * 1000
    else:
        return int(s)

try:
    f = sa.wav_file(sys.argv[1])
except IndexError:
    raise ValueError("Start the script with the file name")

notes = ['A0', 'A#0', 'B0',
        'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1',
        'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
        'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
        'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
        'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
        'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
        'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7',
        'C8']

while True:
    In = input('>>> ')
    In = In.split(' ', 1)
    try:
        if In[0] == 'j':
            try:
                f.jump_to(cast_to_time(In[1]))
            except IndexError:
                error("jump takes one parameter <time>")
            print(f"{f.tell()}ms")
        elif In[0] == 'g':
            ar = sa.get_pitch(f, int(In[1]))
        elif In[0] == 'p':
            fig = plt.figure(None, (32, 8))
            plot = plt.bar(notes, ar)
            fig.savefig("a.svg")
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
        a, b, c = sys.exc_info()
        print(a, b, c)
