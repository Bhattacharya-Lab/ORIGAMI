import torch
from easydict import EasyDict

from .misc import BlackHole


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        # Use default values if beta parameters are not provided
        beta1 = 0.9
        beta2 = 0.999
        if hasattr(cfg, 'beta1'):
            beta1 = cfg.beta1
        if hasattr(cfg, 'beta2'):
            beta2 = cfg.beta2
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(beta1, beta2)
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    # Convert dict to EasyDict if necessary
    if isinstance(cfg, dict) and not isinstance(cfg, EasyDict):
        cfg = EasyDict(cfg)
    
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=getattr(cfg, 'factor', 0.1),
            patience=getattr(cfg, 'patience', 10),
            min_lr=getattr(cfg, 'min_lr', 1e-6),
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=getattr(cfg, 'milestones', [30, 60, 90]),
            gamma=getattr(cfg, 'gamma', 0.1),
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=getattr(cfg, 'gamma', 0.99),
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def log_losses(out, it, tag, logger=BlackHole(), writer=BlackHole(), others={}):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.6f' % out.overall.item()
    for k, v in out.items():
        if k == 'overall': continue
        logstr += ' | loss(%s) %.6f' % (k, v.item())
    for k, v in others.items():
        logstr += ' | %s %.6f' % (k, v)
    logger.info(logstr)

    for k, v in out.items():
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, it)
        else:
            writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ValidationLossTape(object):

    def __init__(self):
        super().__init__()
        self.accumulate = {}
        self.others = {}
        self.total = 0

    def update(self, out, n, others=None):
        self.total += n
        for k, v in out.items():
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        for k, v in others.items():
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()
        

    def log(self, it, logger=BlackHole(), writer=BlackHole()):
        avg = EasyDict({k:v / self.total for k, v in self.accumulate.items()})
        avg_others = EasyDict({k:v / self.total for k, v in self.others.items()})
        log_losses(avg, it, 'val', logger, writer, others=avg_others)
        return avg['overall']
