# np only used in mp3 export so far
# import numpy as np

import os
from pathlib import Path
import glob

import essentia.standard as es
import pretty_midi

from mir_eval.sonify import pitch_contour

SAMPLE_RATE=44100.0
OUT_ROOT='extraction_examples/essentia_melodia'


def extract_from_mp3_to_midi(file_path):
    print('Processing', file_path)

    # Load audio file
    loader = es.EqloudLoader(filename=file_path, sampleRate=SAMPLE_RATE)
    audio_data = loader()
    # print("Duration of the audio sample [sec]:")
    # print(len(audio)/SAMPLE_RATE)

    # Extract melody as pitch values
    # TODO Tweak params
    melody_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, pitch_confidence = melody_extractor(audio_data)


    ######################################################################
    ##### Export as MIDI
    onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio_data)
    # print("MIDI notes:", notes) # Midi pitch number
    # print("MIDI note onsets:", onsets)
    # print("MIDI note durations:", durations)

    instrument = pretty_midi.Instrument(program=0) # = 'Acoustic Grand Piano'

    pm = pretty_midi.PrettyMIDI()
    for pitch, onset, duration in zip(notes, onsets, durations):
        # NOTE velocity = 100 is a default value that is negligible for this scenario
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=onset, end=onset + duration)
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    out_path = f'{OUT_ROOT}/{Path(file_path).stem}.mid'
    pm.write(out_path)

    print('Result written to', out_path)

    ######################################################################



    # ######################################################################
    # ##### Export as raw audio
    # # Pitch is estimated on frames. Compute frame time positions.
    # pitch_times = np.linspace(0.0,len(audio)/SAMPLE_RATE,len(pitch_values) )
    # # Essentia operates with float32 ndarrays instead of float64, so let's cast it.
    # synthesized_melody = pitch_contour(pitch_times, pitch_values, SAMPLE_RATE).astype(np.float32)[:len(audio)]
    # # NOTE Der StereoMuxer mixt Original-Audio und Melodie zusammen
    # es.AudioWriter(filename='extraction_examples/essentia_melodia/' + base_name + '.mp3', format='mp3')(es.StereoMuxer()(audio, synthesized_melody))

    # ######################################################################



def process_folder(folder):
    # input_files=glob.glob(os.path.join('extraction_examples/inputs/', '**/*.mp3'), recursive=True)
    input_files = glob.glob(os.path.join(folder, '**/*.mp3'), recursive=True)
    for file_path in input_files:
        extract_from_mp3_to_midi(file_path)

def main():
    process_folder('extraction_examples/inputs/')

if __name__ == '__main__':
  main()