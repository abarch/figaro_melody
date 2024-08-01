import glob
import os
from pathlib import Path

import essentia.standard as es
import numpy as np
import pretty_midi

from copy import deepcopy

from input_representation import InputRepresentation

SAMPLE_RATE = 44100.0
OUT_ROOT = 'extraction_examples/essentia_melodia'

def extract_from_midi_to_midi(file_path):
    rep = InputRepresentation(file_path)
    # pm_original = pretty_midi.PrettyMIDI(midi_file=file_path)
    pm_original = rep.pm

    # synth = pm_original.fluidsynth().astype(np.float32)  # float32 is essentia's internal datatype; fluidsynth would output float64
    synth = pm_original.synthesize().astype(np.float32)
    eqloud = es.EqualLoudness()  # recommended as preprocessing for melody extraction
    audio_data = eqloud(synth)

    melody_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128, guessUnvoiced=True)
    pitch_values, pitch_confidence = melody_extractor(audio_data)

    # Pitch values to pm
    onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio_data)

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
    # pm_residual = pretty_midi.PrettyMIDI()
    # pm_residual.instruments.extend(pm_original.instruments)
    # pm_residual.time_signature_changes.extend(pm_original.time_signature_changes)
    # pm_residual.resolution = pm_original.resolution
    # pm_residual.key_signature_changes.extend(pm_original.key_signature_changes)
    # rep_residual = InputRepresentation(pm_residual)
    rep_residual = deepcopy(rep)

    # Type: Item
    # Contains start, end, velocity, pitch
    melody_notes = rep_melody.note_items
    residual_notes = rep_residual.note_items

    # TODO MAIN Tweak the thresholds
    time_thresh = 100
    pitch_thresh = 10

    removed_note_idx = 1
    processed_o_notes=0
    removed_notes=0
    condition = lambda m_note, r_note: r_note.instrument != 'drum' and \
        abs(m_note.pitch - r_note.pitch) < pitch_thresh and \
        abs(m_note.start-r_note.start) < time_thresh and \
        abs(m_note.end-r_note.end) < time_thresh
    for m_note_idx, m_note in enumerate(melody_notes):
        for r_note_idx, r_note in enumerate(residual_notes[:]):
            processed_o_notes+=1
            if condition(m_note, r_note):
                # Remove fitting note
                residual_notes.remove(r_note)
                removed_notes+=1
                removed_note_idx = r_note_idx

    print('Processed notes', processed_o_notes)
    print('Removed notes', removed_notes)
    
    print('len(residual_notes) after processing', len(residual_notes))


    # Save to midi file
    out_path_mel = f'{OUT_ROOT}/{Path(file_path).stem}_melody.mid'
    pm_melody.write(out_path_mel)
    print('Melody result written to', out_path_mel)

    out_path_res = f'{OUT_ROOT}/{Path(file_path).stem}_residual.mid'
    rep_residual.pm.write(out_path_res)
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
    # synthesized_melody = pitch_contour(pitch_times, pitch_values, SAMPLE_RATE).astype(np.float32)[:len(audio)]
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
