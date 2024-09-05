
import os
import glob
import pickle
import random
import torch
from torch.utils.data.dataloader import DataLoader
from transformers.models.bert.modeling_bert import BertAttention

from models.vae import VqVaeModule
from constants import MASK_TOKEN
from datasets import MidiDataset, SeqCollator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.getenv('ROOT_DIR', os.path.join(os.getenv('TMPDIR', './temp'), 'lmd_full'))
MAX_N_FILES = int(os.getenv('MAX_N_FILES', '-1'))

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))
if device.type == 'cuda':
  N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())
N_WORKERS = int(N_WORKERS)

LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH', os.path.join(os.getenv('SCRATCH', os.getenv('TMPDIR')), 'latent'))
os.makedirs(LATENT_CACHE_PATH, exist_ok=True)


### Create data loaders ###
midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
if MAX_N_FILES > 0:
  midi_files = midi_files[:MAX_N_FILES]

# Shuffle files for approximate parallelizability
random.shuffle(midi_files)

# -----------------------------------
# from generate.py
def load_old_or_new_checkpoint(model_class, checkpoint):
  # assuming transformers>=4.36.0
  pl_ckpt = torch.load(checkpoint, map_location="cpu")
  kwargs = pl_ckpt['hyper_parameters']
  if 'flavor' in kwargs:
    del kwargs['flavor']
  if 'vae_run' in kwargs:
    del kwargs['vae_run']
  model = model_class(**kwargs)
  state_dict = pl_ckpt['state_dict']
  # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
  state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
  try:
    # succeeds for checkpoints trained with transformers>4.13.0
    model.load_state_dict(state_dict)
  except RuntimeError:
    # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
    config = model.transformer.decoder.bert.config
    for layer in model.transformer.decoder.bert.encoder.layer:
      layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
    model.load_state_dict(state_dict)
  if model_class == VqVaeModule:
    model.cpu()
  model.freeze()
  model.eval()
  return model
# -----------------------------------

VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)
# vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=VAE_CHECKPOINT).to(device)
vae_module = load_old_or_new_checkpoint(VqVaeModule, VAE_CHECKPOINT)
vae_module.eval()
vae_module.freeze()

collator = SeqCollator(context_size=vae_module.context_size)

print('***** PRECOMPUTING LATENT REPRESENTATIONS *****')
print(f'Number of files: {len(midi_files)}')
print(f'Using cache: {LATENT_CACHE_PATH}')
print('***********************************************')

for i, file in enumerate(midi_files):
  print(f"{i:4d}/{len(midi_files)}: {file} ", end='')
  cache_key = os.path.basename(file)
  cache_file = os.path.join(LATENT_CACHE_PATH, cache_key)

  try:
    latents, codes = pickle.load(open(cache_file, 'rb'))
    print(f'(already cached: {len(latents)} bars)')
    continue
  except:
    pass

  ds = MidiDataset([file], vae_module.context_size, 
    description_flavor='none',
    max_bars_per_context=1, 
    bar_token_mask=MASK_TOKEN,
    print_errors=True,
  )

  dl = DataLoader(ds, 
    collate_fn=collator, 
    batch_size=BATCH_SIZE, 
    num_workers=N_WORKERS, 
    pin_memory=True
  )

  latents, codes = [], []
  for batch in dl:
    x = batch['input_ids'].to(device)

    out = vae_module.encode(x)
    latents.append(out['z'])
    codes.append(out['codes'])
  
  if len(latents) == 0:
    continue
    
  latents = torch.cat(latents).cpu()
  codes = torch.cat(codes).cpu()
  print(f'(caching latents: {latents.size(0)} bars)')

  # Try to store the computed representation in the cache directory
  try:
    pickle.dump((latents, codes), open(cache_file, 'wb'))
  except Exception as err:
    print('Unable to cache file:', str(err))