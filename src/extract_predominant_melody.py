import glob
import os
from pathlib import Path

import essentia.standard as es
import numpy as np
import pretty_midi

from copy import deepcopy

SAMPLE_RATE = 44100.0
OUT_ROOT = 'extraction_examples/essentia_melodia'

def extract_from_midi_to_midi(file_path):
    pm_original = pretty_midi.PrettyMIDI(midi_file=file_path)

    # synth = pm_original.fluidsynth().astype(np.float32)  # float32 is essentia's internal datatype; fluidsynth would output float64
    synth = pm_original.synthesize().astype(np.float32)
    eqloud = es.EqualLoudness()  # recommended as preprocessing for melody extraction
    audio_data = eqloud(synth)

    melody_extractor = es.PredominantPitchMelodia(guessUnvoiced=True)
    pitch_values, pitch_confidence = melody_extractor(audio_data)

    # Pitch values to pm
    onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio_data)

    pm_melody = deepcopy(pm_original)
    melody_notes = []
    for pitch, onset, duration in zip(notes, onsets, durations):
        # NOTE velocity = -1 is a default value and is changed later. Otherwise -1 is an invalid velocity value
        note = pretty_midi.Note(velocity=-1, pitch=int(pitch), start=onset, end=onset + duration)
        melody_notes.append(note)

    # Compute residual
    pm_melody.instruments.clear()  # Will be recalculated below

    # pm_residual = deepcopy(pm_original)

    time_thresh_start = 1
    time_thresh_end = 1
    pitch_thresh = 4

    processed_orig_notes = 0
    removed_notes = 0

    # Elements to be deleted from o_notes (= melody notes in original notes)
    def _filter_condition(melody_note, orig_note):
        return abs(melody_note.pitch - orig_note.pitch) <= pitch_thresh and \
                abs(melody_note.start - orig_note.start) <= time_thresh_start and \
                abs(melody_note.end - orig_note.end) <= time_thresh_end

    for instr in pm_original.instruments:
        if not instr.is_drum:
            # Debugging
            processed_orig_notes += len(instr.notes)
            # --------------------------------------

            o_notes_to_delete = []

            new_mel_instrument = pretty_midi.Instrument(name=instr.name, program=instr.program)
            new_mel_instrument.control_changes = instr.control_changes
            new_mel_instrument.pitch_bends = instr.pitch_bends

            # Remember the index of the position the melody note was found in the original song
            # to reduce number of iterations
            found_mel_note_idx = 0
            for m_note_idx, m_note in enumerate(melody_notes):
                mel_instrument_set = False
                mel_note_block_found = False
                # o_note is the original note of the instrument
                for o_note_idx, o_note in enumerate(instr.notes[found_mel_note_idx:], start=found_mel_note_idx):
                    # m_note and o_note are close
                    if _filter_condition(melody_note=m_note, orig_note=o_note):
                        # Collect indices of o_notes that are found in the melody notes
                        o_notes_to_delete.append(o_note_idx)

                        # Only set each melody instrument once for each m_note
                        if not mel_instrument_set:
                            # Remember the current o_note_idx to reduce iterations for next m_note
                            # because we know that the m_notes cannot occur simultaneously.
                            found_mel_note_idx = o_note_idx
                            # Remember whether a note already matched for processing the following o_notes
                            mel_note_block_found = True
                            # Set velocity of melody note according to the matching non-melody note
                            m_note.velocity = o_note.velocity
                            # Add melody note to fitting instrument
                            new_mel_instrument.notes.append(m_note)
                            mel_instrument_set = True
                    # m_note is parted from o_note according to the thresholds
                    else:
                        # if a note already matched and it does not any more,
                        # every o_notes for this m_note have been found
                        # -> Continue with next m_note
                        if mel_note_block_found:
                            break

                # If velocity is still at the default value, take the value of the previous m_note
                if m_note.velocity == -1:
                    if m_note_idx == 0:  # If the current melody note is the first and did not match any o_note:
                        m_note.velocity = instr.notes[0].velocity  # take the first o_note's velocity
                    else:
                        m_note.velocity = melody_notes[m_note_idx - 1].velocity

            # Add the melody notes with the fitting instrument again
            if len(new_mel_instrument.notes) > 0:
                pm_melody.instruments.append(new_mel_instrument)

            instr.notes = np.delete(np.array(instr.notes), o_notes_to_delete)

            # Debug
            removed_notes += len(o_notes_to_delete)

    # Debug output
    print('Processed notes', processed_orig_notes)
    print('Removed notes', removed_notes)

    # Save to midi file
    out_path_mel = f'{OUT_ROOT}/{Path(file_path).stem}_melody.mid'
    pm_melody.write(out_path_mel)
    print('Melody result written to', out_path_mel)

    out_path_res = f'{OUT_ROOT}/{Path(file_path).stem}_residual.mid'
    # Debugging
    # out_path_res = f'{OUT_ROOT}/{Path(file_path).stem}_residual_{time_thresh_start};{time_thresh_end}_{pitch_thresh}.mid'
    pm_original.write(out_path_res)
    print('Residual result written to', out_path_res)

def extract_from_mp3_to_midi(file_path):
    print('Processing', file_path)

    # Load audio file
    loader = es.EqloudLoader(filename=file_path, sampleRate=SAMPLE_RATE)
    audio_data = loader()
    # print("Duration of the audio sample [sec]:")
    # print(len(audio)/SAMPLE_RATE)

    # Extract melody as pitch values
    melody_extractor = es.PredominantPitchMelodia(guessUnvoiced=True)
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
        # Old! Has to be adjusted as in extract_from_midi_to_midi
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
