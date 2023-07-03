from flask import Flask, request, send_file
import spectrum_analyzer as sa
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy
from io import BytesIO
import subprocess
app = Flask(__name__)

wav = None
notes = ['A0', '#A0', 'B0', 'C1', '#C1', 'D1', '#D1', 'E1', 'F1', '#F1', 'G1', '#G1',
        'A1', '#A1', 'B1', 'C2', '#C2', 'D2', '#D2', 'E2', 'F2', '#F2', 'G2', '#G2',
        'A2', '#A2', 'B2', 'C3', '#C3', 'D3', '#D3', 'E3', 'F3', '#F3', 'G3', '#G3',
        'A3', '#A3', 'B3', 'C4', '#C4', 'D4', '#D4', 'E4', 'F4', '#F4', 'G4', '#G4',
        'A4', '#A4', 'B4', 'C5', '#C5', 'D5', '#D5', 'E5', 'F5', '#F5', 'G5', '#G5',
        'A5', '#A5', 'B5', 'C6', '#C6', 'D6', '#D6', 'E6', 'F6', '#F6', 'G6', '#G6',
        'A6', '#A6', 'B6', 'C7', '#C7', 'D7', '#D7', 'E7', 'F7', '#F7', 'G7', '#G7',
        'A7', '#A7', 'B7', 'C8']

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return send_file('ide.html')
    else:
        global wav
        fn = request.form['fn']
        print(fn)
        if fn:
            if fn[-4:] != ".wav":
                subprocess.Popen(['ffmpeg', '-i', fn, '-ac', '1', 'a.wav'])
                wav = sa.wav_file("a.wav")
            else:
                wav = sa.wav_file(fn)
        return ('', 202)

@app.route('/file')
def file():
    wav.jump_to(int(1000 * float(request.args.get('start'))))
    wav.play(int(request.args.get('dura')))
    return 'DONE'

@app.route('/plot')
def plot():
    begin = float(request.args.get('start'))
    dura = int(request.args.get('dura'))
    wav.jump_to(int(1000 * begin))
    ar = wav.get(dura)
    fig = plt.figure(None, (16, 8))
    # we use linspace because shape must match
    plt.plot(numpy.linspace(begin, begin + dura / 1000, len(ar)), ar)
    a_svg = BytesIO()
    fig.savefig(a_svg)
    return a_svg.getvalue()

@app.route('/fitrequest')
def fit():
    begin = request.args.get('start')
    dura = request.args.get('dura')
    wav.jump_to(int(1000 * float(begin)))
    ar = sa.get_pitch(wav, int(dura))
    fig = plt.figure(None, (16, 8))
    ar[ar == 0] = 1
    plt.bar(notes, numpy.log10(ar))
    a_svg = BytesIO()
    fig.savefig(a_svg)
    return a_svg.getvalue()
