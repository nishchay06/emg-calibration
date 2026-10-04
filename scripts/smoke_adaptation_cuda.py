"""One synthetic upstream CUDA update; not a CER reproduction."""
import json
import sys
import time
from pathlib import Path

project = Path(sys.argv[1])
upstream = Path(sys.argv[2])
sys.path.insert(0, str(project / 'scripts'))
sys.path.insert(0, str(upstream))
import adapt
import numpy as np
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from emg2qwerty import transforms
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.lightning import TDSConvCTCModule

assert torch.cuda.is_available()
pl.seed_everything(1501, workers=True)
args = adapt.arguments(['--check-config', '--user', 'user0', '--upstream-dir', str(upstream),
                        '--data-dir', '/workspace/data', '--output-dir', '/workspace/unused-smoke-plan',
                        '--checkpoint', '/workspace/emg2qwerty/models/generic.ckpt'])
config = adapt.compose_config(args)
cfg = OmegaConf.create(config)
assert adapt.sha256(args.checkpoint) == adapt.GENERIC_SHA256
module = TDSConvCTCModule.load_from_checkpoint(str(args.checkpoint), map_location='cpu',
    optimizer=cfg.optimizer, lr_scheduler=cfg.lr_scheduler, decoder=cfg.decoder)
initial = module.model[4].weight.detach().clone()
transform = transforms.Compose([instantiate(item) for item in cfg.transforms.train])
rng = np.random.default_rng(1501)
samples = []
for _ in range(2):
    raw = np.zeros(10000, dtype=[('emg_left', np.float32, (16,)),
        ('emg_right', np.float32, (16,)), ('time', np.float64)])
    raw['emg_left'] = rng.standard_normal((10000, 16), dtype=np.float32)
    raw['emg_right'] = rng.standard_normal((10000, 16), dtype=np.float32)
    raw['time'] = np.arange(10000) / 2000
    samples.append((transform(raw), torch.tensor(LabelData.from_str('emgtest').labels)))
loader = DataLoader(samples, batch_size=2, num_workers=0, collate_fn=WindowedEMGDataset.collate)
evidence = {'phase': 'synthetic-cuda-smoke', 'real_data_used': False,
    'released_checkpoint_used': True, 'gpu': torch.cuda.get_device_name(),
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'checks': {}}
class Capture(pl.Callback):
    def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        evidence['checks']['finite_gradients'] = True
        evidence['learning_rate'] = optimizer.param_groups[0]['lr']
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        evidence['loss'] = float(outputs['loss'].detach())
        assert np.isfinite(evidence['loss'])
        evidence['checks']['finite_loss'] = True
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=150, max_steps=1,
    callbacks=[Capture()], logger=False, enable_checkpointing=False,
    enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
start = time.monotonic()
trainer.fit(module, train_dataloaders=loader)
evidence['fit_seconds'] = time.monotonic() - start
evidence['optimizer_steps'] = trainer.global_step
evidence['changed_head_elements'] = int(torch.count_nonzero(module.model[4].weight.detach().cpu() != initial))
assert trainer.global_step == 1 and evidence['changed_head_elements'] > 0
evidence['checks']['weights_updated'] = True
assert evidence['learning_rate'] == 1e-8
evidence['passed'] = all(evidence['checks'].values())
print('CUDA_SMOKE_RESULT=' + json.dumps(evidence, sort_keys=True))
