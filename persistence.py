import os
from mxnet import gluon, init

def restore_block(block, param_file, ctx):
    if os.path.isfile(param_file):
        block.load_params(param_file, ctx)
        return True
    return False

class TrainingSession:
    def __init__(self, ckpt_dir='ckpt'):
        os.makedirs(ckpt_dir, exist_ok=True)
        self._cpdir = ckpt_dir
        self._blocks = {} # dict of key -> [block, lr]
        self._trainers = {} # dict of key -> trainer

    def add_block(self, key, block, lr=0.01):
        self._blocks[key] = [block, lr]

    def init_all(self, ctx):
        for key, [block, lr] in self._blocks.items():
            self._init_block(block, self._get_path(key), ctx)
            self._trainers[key] = self._trainer(block, self._get_path(key, 'trainer'), lr)

    def get_block(self, key):
        return self._blocks[key][0]

    def get_trainer(self, key):
        return self._trainers[key]

    def save_all(self):
        for key, [block, _] in self._blocks.items():
            block.save_params(self._get_path(key))
        for key, trainer in self._trainers.items():
            trainer.save_states(self._get_path(key, 'trainer'))

    def _get_path(self, key, typ='params'):
        return os.path.join(self._cpdir, '{}.{}'.format(key, typ))

    def _init_block(self, block, param_file, ctx):
        if not restore_block(block, param_file, ctx):
            xavinit = init.Xavier(magnitude=2.24)
            block.initialize(xavinit, ctx)

    def _trainer(self, block, state_file, lr):
        trainer = gluon.Trainer(block.collect_params(), 'adam', {'learning_rate': lr})
        if os.path.isfile(state_file):
            trainer.load_states(state_file)
        return trainer
