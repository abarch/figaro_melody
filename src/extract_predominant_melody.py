import argparse
import glob
import os
from pathlib import Path
import warnings

import essentia.standard as es
import numpy as np
import pretty_midi

import mido

import time

from copy import deepcopy

SAMPLE_RATE = 44100.0

def test_synthesis(file_path):
  pm = pretty_midi.PrettyMIDI(file_path)

  # Loudness equalization
  eqloud = es.EqualLoudness()

  synthed_synthesize = pm.synthesize().astype(np.float32)
  writer = es.MonoWriter(filename='temp/synthtest/eq_synthesize.wav', sampleRate=SAMPLE_RATE, format='wav')
  writer(eqloud(synthed_synthesize))

  synthed_fluidsynth = pm.fluidsynth().astype(np.float32)
  writer = es.MonoWriter(filename='temp/synthtest/eq_fluidsynth.wav', sampleRate=SAMPLE_RATE, format='wav')
  writer(eqloud(synthed_fluidsynth))

def extract_from_midi_to_midi(file_path, output):
  """
  Extracts the melody of a midi file. Stores the melody and accompaniment as separate files
  :param file_path: Input midi file
  :param output: The base path to store the files in
  :return: None
  """

  try:
    pm_original = pretty_midi.PrettyMIDI(midi_file=file_path)
    pm_original.remove_invalid_notes()
  except (OSError, EOFError, ValueError, KeyError, mido.midifiles.meta.KeySignatureError) as err:
    print(f'ERROR Corrupt midi file {file_path}. Skipping it.')
    print('Cause of corruption:', str(err))
  else:
    # Productive
    melody_notes = compute_melody_notes(pm_original)
    # Copies all time_signature changes, etc. from the original midi file
    pm_melody = deepcopy(pm_original)

    # Testing
    # pm_melody = pretty_midi.PrettyMIDI('input_melody.mid')
    # melody_notes = []
    # for instr in pm_melody.instruments:  # Flatten as single Instrument. # NOTE Can lead to duplicate notes(!)
    #   melody_notes.extend(instr.notes)

    # Compute accompaniment
    pm_melody.instruments.clear()  # Will be recalculated below

    time_thresh_start = 2
    time_thresh_end = 2
    pitch_thresh = 3  # can (and should) stay small because values are quantized!

    processed_orig_notes = 0
    removed_notes = 0

    # Elements to be deleted from o_notes (= melody notes in original notes)
    def _filter_condition(melody_note, orig_note):
      """
      Checks whether melody and original note are close to each other in terms of times and pitch
      :param melody_note: Melody note
      :param orig_note: Original note
      :return: True if close, False if not
      """
      return abs(melody_note.pitch - orig_note.pitch) <= pitch_thresh and \
             abs(melody_note.start - orig_note.start) <= time_thresh_start and \
             abs(melody_note.end - orig_note.end) <= time_thresh_end

    # Debugging:
    # Tracks how many melody notes have been assigned to an original note with bools for each idx
    m_n_assigned = np.zeros(len(melody_notes), dtype=bool)
    #------------------------------------------------------
    for instr in pm_original.instruments:
      if not instr.is_drum:
        # Debugging
        processed_orig_notes += len(instr.notes)
        # --------------------------------------

        o_notes_to_delete = set([])

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
          for o_note_idx, o_note in enumerate(instr.notes[:]):
          # for o_note_idx, o_note in enumerate(instr.notes[found_mel_note_idx:], start=found_mel_note_idx):
            # m_note and o_note are close
            if _filter_condition(melody_note=m_note, orig_note=o_note):
              # Collect indices of o_notes that are found in the melody notes
              o_notes_to_delete.add(o_note_idx)  # wrap in if last element != o_note_idx
              m_n_assigned[m_note_idx] = True
              # Debug: List of actual differences:
              # pitch_diff = abs(m_note.pitch - o_note.pitch)
              # start_diff = abs(m_note.start - o_note.start)
              # end_diff = abs(m_note.end - o_note.end)

              # print(f'pitch_diff: {pitch_diff} start_diff: {start_diff} end_diff: {end_diff}')
              # Only set each melody instrument once for each m_note
              if not mel_instrument_set:
                # Remember the current o_note_idx to reduce iterations for next m_note
                # because we know that the m_notes cannot occur simultaneously.
                found_mel_note_idx = o_note_idx
                # Remember whether a note already matched for processing the following o_notes
                mel_note_block_found = True
                # Set velocity of melody note according to the matching non-melody note
                m_note.velocity = o_note.velocity

                m_note.start, m_note.end = o_note.start, o_note.end
                m_note.pitch = o_note.pitch

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
          pm_melody.remove_invalid_notes()  # optional
          pm_melody.instruments.append(new_mel_instrument)

        instr.notes = np.delete(np.array(instr.notes), list(o_notes_to_delete))

        # Debug
        removed_notes += len(o_notes_to_delete)

    # Debug output
    print('Processed notes', processed_orig_notes)
    print('Removed notes', removed_notes)

    print('Unassigned m_notes:', len(np.argwhere(m_n_assigned == False)))

    # With clean_midi (lmd) as input:
    p = Path(file_path)
    nfile_path = f'{p.parts[-2]} - {p.parts[-1]}'  # with clean_midi (lmd) as input

    # Save to midi file
    out_path_mel = os.path.join(output, f'{Path(nfile_path).stem}_melody.mid')
    pm_melody.write(out_path_mel)
    print('Melody result written to', out_path_mel)

    out_path_accomp = os.path.join(output, f'{Path(nfile_path).stem}_accompaniment.mid')
    # Debugging
    # out_path_accomp = os.path.join(output, f'{Path(file_path).stem}_accompaniment_{time_thresh_start};{time_thresh_end}_{pitch_thresh}.mid')
    pm_original.write(out_path_accomp)
    print('Accompaniment result written to', out_path_accomp)


def compute_melody_notes(pm_original):
  """
  Uses MELODIA and pitch contour segmentation to compute the midi notes from pretty_midi object
  :param pm_original: The midi object
  :return: midi melody notes
  :rtype: list[pretty_midi.Note]
  """

  # synth = pm_original.fluidsynth().astype(np.float32)  # float32 is essentia's internal datatype; fluidsynth would output float64
  synth = pm_original.synthesize().astype(np.float32)
  eqloud = es.EqualLoudness()  # recommended as preprocessing for melody extraction
  audio_data = eqloud(synth)
  melody_extractor = es.PredominantPitchMelodia(guessUnvoiced=True)
  pitch_values, pitch_confidence = melody_extractor(audio_data)  # pitch values from unvoiced segments are negative

  # Note that PitchContourSegmentation already returns the quantized midi note numbers, not the physical pitches
  # returns onset times in seconds, already quantized to the equal tempered scale
  # Pitch values to pm
  onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(list(np.absolute(pitch_values)), audio_data)
  melody_notes = []
  for pitch, onset, duration in zip(notes, onsets, durations):
    # NOTE velocity = -1 is a default value in this context and is changed later. Otherwise -1 is an invalid velocity value
    note = pretty_midi.Note(velocity=-1, pitch=round(pitch), start=onset, end=onset + duration)
    melody_notes.append(note)
  return melody_notes


def process_folder(folder, output):
  # input_files = glob.glob(os.path.join(folder, '**/*.mp3'), recursive=True)
  input_files = glob.glob(os.path.join(folder, '**/*.mid'), recursive=True)
  os.makedirs(output, exist_ok=True)
  for file_path in input_files:
    # extract_from_mp3_to_midi(file_path)
    # Assuming that _melody.mid is also present when _accompaniment.mid is
    
    # Converts folder structures like 'artist/song.mid' to 'artist - song.mid'
    p = Path(file_path)
    planned_outpath = f'{p.parts[-2]} - {p.parts[-1]}'  # with clean_midi (lmd) as input
    possibly_existing_outfile_path = os.path.join(output, f'{Path(planned_outpath).stem}_accompaniment.mid')
    # breakpoint()
    if not os.path.exists(possibly_existing_outfile_path):
      # if not file_path in ('../lmd_full/e/e3de19fed976d515437604f7ac0d34a4.mid', '../lmd_full/f/fd93b44a6e334ec2d9ead40a92dcca52.mid', '../lmd_full/e/e363d6d94eb8ad76fa6694f31ec6c79d.mid', '../lmd_full/0/069a3ce3c45b7e15f46138c5e5a67469.mid', '../lmd_full/f/f94b0d40a3388bf932504d1906bec35e.mid', '../lmd_full/e/e7999e9c714f39adfbd9775794daa46b.mid', '../lmd_full/0/0531e73378d4fc4f86ffad4bce283f5a.mid', '../lmd_full/e/e2c42da8bccf9c8067d6bc3af9b957e6.mid', '../lmd_full/0/00a6fcf5afc65dee2a833291b4c11b0c.mid'):
      # if not file_path in ('../clean_midi/U2/The Electric Co..mid'):
      print('Processing', file_path)
      if len(planned_outpath) < 210:  # Prevent error for output filename being too long upon saving the result
        try:
          extract_from_midi_to_midi(file_path, output)
        except RuntimeWarning as w:
          print(f'Skipping {file_path}. RuntimeWarning encountered:', w)
      else:
        print(f'Skipping {file_path}. Output name presumably too long. Len:', len(planned_outpath))
    else:
      print(f'Skipping {file_path}. Already in {output}')


def main():
  # Convert RuntimeWarnings to errors so the can be caught in process_folder
  warnings.simplefilter(action='error', category=RuntimeWarning)
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()

  start_time = time.time()

  process_folder(folder=args.input, output=args.output)

  end_time = time.time()

  print('Execution time (in s):', end_time - start_time)

if __name__ == '__main__':
  main()
