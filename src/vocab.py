import pretty_midi
from collections import Counter
import torchtext
from torch import Tensor

from constants import (
  DEFAULT_VELOCITY_BINS,
  DEFAULT_DURATION_BINS,
  DEFAULT_TEMPO_BINS,
  DEFAULT_POS_PER_QUARTER,
  DEFAULT_NOTE_DENSITY_BINS,
  DEFAULT_MEAN_VELOCITY_BINS,
  DEFAULT_MEAN_PITCH_BINS,
  DEFAULT_MEAN_DURATION_BINS
)


from constants import (
  MAX_BAR_LENGTH,
  MAX_N_BARS,

  PAD_TOKEN,
  UNK_TOKEN,
  BOS_TOKEN,
  EOS_TOKEN,
  MASK_TOKEN,

  TIME_SIGNATURE_KEY,
  BAR_KEY,
  POSITION_KEY,
  INSTRUMENT_KEY,
  PITCH_KEY,
  VELOCITY_KEY,
  DURATION_KEY,
  TEMPO_KEY,
  CHORD_KEY,

  NOTE_DENSITY_KEY,
  MEAN_PITCH_KEY,
  MEAN_VELOCITY_KEY,
  MEAN_DURATION_KEY,
  MELODY_NOTE_KEY,
  MELODY_INSTRUMENT_KEY
)

class Tokens:
  def get_instrument_tokens(key=INSTRUMENT_KEY, with_drum=True):
    tokens = [f'{key}_{pretty_midi.program_to_instrument_name(i)}' for i in range(128)]
    if with_drum:
      tokens.append(f'{key}_drum')
    return tokens

  def get_chord_tokens(key=CHORD_KEY, qualities = ['maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'None']):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    chords = [f'{root}:{quality}' for root in pitch_classes for quality in qualities]
    chords.append('N:N')

    tokens = [f'{key}_{chord}' for chord in chords]
    return tokens

  def get_time_signature_tokens(key=TIME_SIGNATURE_KEY):
    denominators = [2, 4, 8, 16]
    time_sigs = [f'{p}/{q}' for q in denominators for p in range(1, MAX_BAR_LENGTH*q + 1)]
    tokens = [f'{key}_{time_sig}' for time_sig in time_sigs]
    return tokens

  def get_midi_tokens(
    instrument_key=INSTRUMENT_KEY, 
    time_signature_key=TIME_SIGNATURE_KEY,
    pitch_key=PITCH_KEY,
    velocity_key=VELOCITY_KEY,
    duration_key=DURATION_KEY,
    tempo_key=TEMPO_KEY,
    bar_key=BAR_KEY,
    position_key=POSITION_KEY
  ):
    instrument_tokens = Tokens.get_instrument_tokens(instrument_key)

    pitch_tokens = [f'{pitch_key}_{i}' for i in range(128)] + [f'{pitch_key}_drum_{i}' for i in range(128)]
    velocity_tokens = [f'{velocity_key}_{i}' for i in range(len(DEFAULT_VELOCITY_BINS))]
    duration_tokens = [f'{duration_key}_{i}' for i in range(len(DEFAULT_DURATION_BINS))]
    tempo_tokens = [f'{tempo_key}_{i}' for i in range(len(DEFAULT_TEMPO_BINS))]
    bar_tokens = [f'{bar_key}_{i}' for i in range(MAX_N_BARS)]
    position_tokens = [f'{position_key}_{i}' for i in range(MAX_BAR_LENGTH*4*DEFAULT_POS_PER_QUARTER)]

    time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)

    return (
      time_sig_tokens +
      tempo_tokens + 
      instrument_tokens + 
      pitch_tokens + 
      velocity_tokens + 
      duration_tokens + 
      bar_tokens + 
      position_tokens
    )

  def get_melody_tokens(melody_note_key=MELODY_NOTE_KEY, melody_instrument_key=MELODY_INSTRUMENT_KEY):
    # Shape: Melody Instrument_(instrument)
    instrument_tokens = Tokens.get_instrument_tokens(key=melody_instrument_key, with_drum=False)

    # Shape: Melody Note_(pitch);(velocity);(duration)
    pitch_values = list(range(128))
    velocity_values = list(range(len(DEFAULT_VELOCITY_BINS)))
    duration_values = list(range(len(DEFAULT_DURATION_BINS)))
    note_tokens = [f'{melody_note_key}_{p};{v};{d}' for p in pitch_values for v in velocity_values for d in duration_values]
    return instrument_tokens + note_tokens

class Vocab:
  def __init__(self, counter, specials=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN], unk_token=UNK_TOKEN):
    self.vocab = torchtext.vocab.vocab(counter)

    self.specials = specials
    for i, token in enumerate(self.specials):
      self.vocab.insert_token(token, i)
    
    if unk_token in specials:
      self.vocab.set_default_index(self.vocab.get_stoi()[unk_token])

  def to_i(self, token):
    return self.vocab.get_stoi()[token]

  def to_s(self, idx):
    if idx >= len(self.vocab):
      return UNK_TOKEN
    else:
      return self.vocab.get_itos()[idx]

  def __len__(self):
    return len(self.vocab)

  def encode(self, seq):
    return self.vocab(seq)

  def decode(self, seq):
    if isinstance(seq, Tensor):
      seq = seq.numpy()
    return self.vocab.lookup_tokens(seq)


class RemiVocab(Vocab):
  def __init__(self):
    midi_tokens = Tokens.get_midi_tokens()
    chord_tokens = Tokens.get_chord_tokens()

    self.tokens = midi_tokens + chord_tokens

    counter = Counter(self.tokens)
    super().__init__(counter)


class DescriptionVocab(Vocab):
  def __init__(self, add_melody_tokens=False):
    time_sig_tokens = Tokens.get_time_signature_tokens()
    instrument_tokens = Tokens.get_instrument_tokens()
    chord_tokens = Tokens.get_chord_tokens()

    bar_tokens = [f'Bar_{i}' for i in range(MAX_N_BARS)]
    density_tokens = [f'{NOTE_DENSITY_KEY}_{i}' for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
    velocity_tokens = [f'{MEAN_VELOCITY_KEY}_{i}' for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]
    pitch_tokens = [f'{MEAN_PITCH_KEY}_{i}' for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
    duration_tokens = [f'{MEAN_DURATION_KEY}_{i}' for i in range(len(DEFAULT_MEAN_DURATION_BINS))]

    self.tokens = (
      time_sig_tokens +
      instrument_tokens +
      chord_tokens +
      density_tokens +
      velocity_tokens +
      pitch_tokens +
      duration_tokens +
      bar_tokens
    )

    # Conditionally add melody tokens
    if add_melody_tokens:
      melody_tokens = Tokens.get_melody_tokens()
      self.tokens += melody_tokens

    counter = Counter(self.tokens)
    super().__init__(counter)
