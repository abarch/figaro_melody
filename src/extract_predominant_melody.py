import glob
import os
from pathlib import Path

import essentia.standard as es
import numpy as np
import pretty_midi

from input_representation import InputRepresentation

SAMPLE_RATE = 44100.0
OUT_ROOT = 'extraction_examples/essentia_melodia'

def extract_from_midi_to_midi(file_path):
    rep = InputRepresentation(file_path)
    # pm_original = pretty_midi.PrettyMIDI(midi_file=file_path)
    pm_original = rep.pm

    synth = pm_original.fluidsynth().astype(np.float32)  # float32 is essentia's internal datatype; fluidsynth would output float64
    eqloud = es.EqualLoudness()  # recommended as preprocessing for melody extraction
    audio_data = eqloud(synth)  # Input für essentia

    melody_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, pitch_confidence = melody_extractor(audio_data)

    # Pitch values to pm
    onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio_data)
    # print("MIDI notes:", notes) # Midi pitch number

    instrument = pretty_midi.Instrument(program=0)  # = 'Acoustic Grand Piano'
    pm_melody = pretty_midi.PrettyMIDI()
    for pitch, onset, duration in zip(notes, onsets, durations):
        # NOTE velocity = 100 is a default value that is negligible for this scenario
        # TODO use velocity from input if input is midi -> add input representation as parameter!
        # Velocity ist in note_items gespeichert
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=onset, end=onset + duration)
        instrument.notes.append(note)
    pm_melody.instruments.append(instrument)

    rep_melody = InputRepresentation(pm_melody)

    # Compute residual


    # Copy original pm
    pm_residual = pretty_midi.PrettyMIDI()
    pm_residual.instruments.extend(pm_original.instruments)
    pm_residual.time_signature_changes.extend(pm_original.time_signature_changes)
    pm_residual.resolution = pm_original.resolution
    pm_residual.key_signature_changes.extend(pm_original.key_signature_changes)
    rep_residual = InputRepresentation(pm_residual)

    # Type: Item
    # Contains start, end, velocity, pitch
    melody_notes = rep_melody.note_items
    original_notes = rep_residual.note_items

    # TODO MAIN Tweak the thresholds
    time_thresh = 1
    pitch_thresh = 1

    removed_note_idx = 0
    for m_note_idx in range(len(melody_notes)):
        for o_note_idx in range(removed_note_idx, len(original_notes)):
            m_note = melody_notes[m_note_idx]
            o_note = original_notes[o_note_idx]
            # Assumes that there is only one track
            if abs(m_note.pitch - o_note.pitch) < pitch_thresh and (m_note.start-o_note.start) < time_thresh:
                # Remove fitting note
                o_note.pitch = 0
                removed_note_idx = o_note_idx
                break

    # Save to midi file
    out_path_mel = f'{OUT_ROOT}/{Path(file_path).stem}_melody.mid'
    pm_melody.write(out_path_mel)
    print('Melody result written to', out_path_mel)

    out_path_res = f'{OUT_ROOT}/{Path(file_path).stem}_residual.mid'
    pm_residual.write(out_path_res)
    print('Residual result written to', out_path_res)

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

    export_to_midi(audio_data, file_path, pitch_values)

    # ######################################################################
    # ##### Export as raw audio
    # # Pitch is estimated on frames. Compute frame time positions.
    # pitch_times = np.linspace(0.0,len(audio)/SAMPLE_RATE,len(pitch_values) )
    # # Essentia operates with float32 ndarrays instead of float64, so let's cast it.
    # synthesized_melody = pitch_contour(pitch_times, pitch_values, SAMPLE_RATE).astype(np.float32)[:len(audio)]
    # # NOTE Der StereoMuxer mixt Original-Audio und Melodie zusammen
    # es.AudioWriter(filename='extraction_examples/essentia_melodia/' + base_name + '.mp3', format='mp3')(es.StereoMuxer()(audio, synthesized_melody))
    # ######################################################################

def export_to_midi(audio_data, file_path, pitch_values):
    onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio_data)
    # print("MIDI notes:", notes) # Midi pitch number
    # print("MIDI note onsets:", onsets)
    # print("MIDI note durations:", durations)
    instrument = pretty_midi.Instrument(program=0)  # = 'Acoustic Grand Piano'
    pm_melody = pretty_midi.PrettyMIDI()
    for pitch, onset, duration in zip(notes, onsets, durations):
        # NOTE velocity = 100 is a default value that is negligible for this scenario
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=onset, end=onset + duration)
        instrument.notes.append(note)
    pm_melody.instruments.append(instrument)

    out_path = f'{OUT_ROOT}/{Path(file_path).stem}_melody.mid'
    pm_melody.write(out_path)
    print('Result written to', out_path)

def process_folder(folder):
    # input_files = glob.glob(os.path.join(folder, '**/*.mp3'), recursive=True)
    input_files = glob.glob(os.path.join(folder, '**/*.mid'), recursive=True)
    for file_path in input_files:
        # extract_from_mp3_to_midi(file_path)
        extract_from_midi_to_midi(file_path)

def main():
    process_folder('extraction_examples/inputs/')

if __name__ == '__main__':
  main()