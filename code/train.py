"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def zeropower_via_newtonschulz5(g, steps=5):
    """Orthogonalize a 2D update with the quintic Newton-Schulz iteration, evaluated in bfloat16.

    Coefficients (3.4445, -4.7750, 2.0315) maximize the slope at zero, so a handful of iterations
    yields US'V^T with S' close to uniform rather than exactly UV^T, which costs no accuracy.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum whose Nesterov-updated gradient is replaced by its orthogonalization.

    Only 2D hidden matrices belong here: embeddings, gains, biases and any untied output head stay
    on AdamW. Weight decay is decoupled, exactly as in AdamW.
    """
    def __init__(self, params, lr=.02, momentum=.95, weight_decay=.1, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                momentum = state['momentum_buffer']
                momentum.lerp_(p.grad, 1 - group['momentum'])
                update = p.grad.lerp_(momentum, group['momentum'])
                update = zeropower_via_newtonschulz5(update, steps=group['ns_steps'])
                update = update * max(1, update.size(-2) / update.size(-1))**.5
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape), alpha=-group['lr'])
        return loss


def split_muon_parameters(model):
    """Muon takes the 2D hidden matrices; embeddings, gains, biases and the output head stay on AdamW."""
    hidden, other = [], []
    head = getattr(model, 'head', None)
    for name, parameter in model.named_parameters():
        module = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
        if parameter.ndim == 2 and isinstance(module, torch.nn.Linear) and module is not head:
            hidden.append(parameter)
        else:
            other.append(parameter)
    return hidden, other


def build_optimizers(model, args):
    """Return (optimizers, peak learning rates); main() applies the shared warmup+cosine shape."""
    if args.optimizer == 'adamw':
        return [torch.optim.AdamW(model.parameters(), lr=args.adamw_lr,
                                  weight_decay=args.weight_decay)], [args.adamw_lr]
    hidden, other = split_muon_parameters(model)
    return ([torch.optim.AdamW(other, lr=args.adamw_lr, weight_decay=args.weight_decay),
             Muon(hidden, lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=args.weight_decay)],
            [args.adamw_lr, args.muon_lr])


def load_teachers(paths, device):
    """Rebuild frozen teachers from checkpoints, keeping path/hash provenance for the run record."""
    teachers, provenance = [], []
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['protocol'] != PROTOCOL:
            raise ValueError('Teacher checkpoint belongs to a different course protocol.')
        teacher, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
        teacher.load_state_dict(checkpoint['model'])
        teacher.requires_grad_(False)
        teacher.eval()
        teachers.append(teacher)
        provenance.append({'checkpoint': str(path), 'checkpoint_sha256': sha(path),
                           'implementation_sha256': implementation_sha,
                           'config': checkpoint['config'], 'seed': checkpoint['seed']})
    return teachers, provenance


def distillation_loss(logits, teachers, ids, hard_loss, temperature, alpha, device):
    """alpha * T^2 * KL(student/T || teacher/T) + (1 - alpha) * hard cross-entropy, per token.

    One teacher contributes its temperature-scaled softmax; several contribute the renormalized
    geometric mean of their probabilities, which is what averaging their log-probabilities means.
    """
    student = F.log_softmax(logits.float() / temperature, dim=-1)
    teacher_precision = 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    with torch.no_grad(), autocast(device, teacher_precision):
        logps = [F.log_softmax(teacher(ids).float() / temperature, dim=-1) for teacher in teachers]
    target = logps[0] if len(logps) == 1 else F.log_softmax(torch.stack(logps).mean(0), dim=-1)
    soft = F.kl_div(student.flatten(0, 1), target.flatten(0, 1), reduction='batchmean', log_target=True)
    return alpha * temperature ** 2 * soft + (1 - alpha) * hard_loss


def wsd_multiplier(step, steps, cooldown_frac, warmup=100):
    """Warmup-stable-decay shape: linear warmup, constant plateau, then a linear cooldown.

    The plateau runs to (1 - cooldown_frac) * steps and is exactly at the peak on its last step;
    the cooldown reaches exactly a tenth of the peak on the final step, matching where the cosine
    schedule ends. The warmup is the same 100-step ramp the cosine schedule uses.
    """
    warmup_factor = min(1., (step+1)/warmup)
    plateau_end = (1.-cooldown_frac)*steps
    if step < plateau_end:
        return warmup_factor
    cooldown_steps = max(1., steps-1-plateau_end)
    return warmup_factor * (1.-.9*min(1., (step-plateau_end)/cooldown_steps))


def multi_token_loss(mtp_logits, ids, weight):
    """Mean over auxiliary heads of the cross-entropy for the token each head predicts.

    Head j predicts ids[:, t+j+2] from position t, so it has ids.shape[1]-(j+2) usable positions.
    Every head is averaged over its own span before the heads are averaged, which keeps the term
    comparable across heads and across window lengths.
    """
    terms = [F.cross_entropy(logits[:, :logits.shape[1]-(index+2)].flatten(0, 1).float(),
                             ids[:, index+2:logits.shape[1]].flatten())
             for index, logits in enumerate(mtp_logits)]
    return weight * torch.stack(terms).mean()


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--optimizer', choices=['adamw','muon'], default='adamw',
                   help='adamw keeps the baseline recipe; muon adds Newton-Schulz updates for 2D hidden matrices.')
    p.add_argument('--adamw-lr', type=float, default=.001,
                   help='Peak AdamW learning rate (baseline value); also the rate for the Muon run tail.')
    p.add_argument('--muon-lr', type=float, default=.02, help='Peak Muon learning rate.')
    p.add_argument('--muon-momentum', type=float, default=.95)
    p.add_argument('--distill-checkpoint', default='',
                   help='Comma-separated teacher checkpoint(s) to distil from; empty disables distillation.')
    p.add_argument('--distill-T', type=float, default=2.0, help='Distillation temperature.')
    p.add_argument('--distill-alpha', type=float, default=.7,
                   help='Weight of the soft KL term; the hard cross-entropy keeps 1 - alpha.')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='Decay of an exponential moving average of the weights; 0 disables it.')
    p.add_argument('--mtp-weight', type=float, default=0.,
                   help='Weight of the auxiliary multi-token prediction loss; needs a model built '
                        'with mtp_k > 0, and 0 disables the term entirely.')
    p.add_argument('--weight-decay', type=float, default=.1,
                   help='Decoupled weight decay for both optimizers; the baseline value is 0.1.')
    p.add_argument('--schedule', choices=['cosine', 'wsd'], default='cosine',
                   help='Learning-rate shape: the original warmup+cosine, or warmup-stable-decay.')
    p.add_argument('--cooldown-frac', type=float, default=.2,
                   help='Fraction of a WSD schedule spent in the linear cooldown.')
    p.add_argument('--epoch-shuffle', action='store_true',
                   help='Draw window starts without replacement, reshuffling the training stream once '
                        'a full pass is used up; off keeps the with-replacement draws.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--save-every', type=int, default=0,
                   help='Overwrite the run checkpoint every N steps, so a long run leaves usable '
                        'weights behind if it is interrupted; 0 saves only at the end.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if not 0. <= args.distill_alpha <= 1.:
        p.error('Distillation alpha must lie in [0, 1].')
    if not 0. <= args.ema_decay < 1.:
        p.error('EMA decay must lie in [0, 1).')
    if not 0. <= args.cooldown_frac < 1.:
        p.error('Cooldown fraction must lie in [0, 1).')
    if args.mtp_weight < 0.:
        p.error('The MTP weight must be non-negative.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    teacher_paths = [Path(item) for item in args.distill_checkpoint.split(',') if item.strip()]
    teachers, teacher_provenance = load_teachers(teacher_paths, device) if teacher_paths else ([], [])
    averaging = ({name: parameter.detach().clone()
                  for name, parameter in model.named_parameters(remove_duplicate=False)}
                 if args.ema_decay > 0 else None)
    mtp_active = args.mtp_weight > 0 and len(getattr(model, 'mtp_heads', ())) > 0
    if args.mtp_weight > 0 and not mtp_active:
        p.error('--mtp-weight needs a model whose config sets mtp_k > 0.')
    optimizers, peak_lrs = build_optimizers(model, args)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    order, cursor = torch.empty(0, dtype=torch.long), 0
    for step in range(args.steps):
        if args.epoch_shuffle:
            # One pass over every legal window start, in random order, then reshuffle; a trailing
            # remainder shorter than a batch is dropped rather than mixed into the next pass.
            if cursor + args.batch_size > len(order):
                order = torch.randperm(len(tokens)-257, generator=rng)
                cursor = 0
            starts = order[cursor:cursor+args.batch_size].to(device)
            cursor += args.batch_size
        else:
            starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        # One shared warmup + cosine-to-10% shape, with a separate peak for each parameter group.
        for optimizer, peak_lr in zip(optimizers, peak_lrs):
            if args.schedule == 'wsd':
                learning_rate = peak_lr * wsd_multiplier(step, args.steps, args.cooldown_frac)
            else:
                learning_rate = peak_lr * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            if mtp_active:
                logits, mtp_logits = model.forward_with_mtp(batch[:,:-1])
            else:
                logits, mtp_logits = model(batch[:,:-1]), []
            loss = F.cross_entropy(logits.flatten(0,1).float(),batch[:,1:].flatten())
            if teachers:
                loss = distillation_loss(logits, teachers, batch[:,:-1], loss,
                                         args.distill_T, args.distill_alpha, device)
            if mtp_logits:
                loss = loss + multi_token_loss(mtp_logits, batch[:,:-1], args.mtp_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for optimizer in optimizers:
            optimizer.step()
        if averaging is not None:
            with torch.no_grad():
                for name, parameter in model.named_parameters(remove_duplicate=False):
                    averaging[name].mul_(args.ema_decay).add_(parameter, alpha=1-args.ema_decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
        if args.save_every > 0 and (step+1)%args.save_every == 0:
            # A CPU copy of the current weights, so an interrupted long run still leaves a usable
            # checkpoint behind; the model itself stays on the training device. The rolling file is
            # overwritten each time, and a per-point snapshot is kept next to it so a rebound can be
            # stepped back past later (checkpoint-step<N>.pt, plus -ema.pt when averaging is on).
            record = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                      'model':{name:value.detach().cpu() for name,value in model.state_dict().items()},
                      'seed':args.seed,'train_tokens':(step+1)*args.batch_size*256}
            torch.save(record, args.run_dir/'checkpoint.pt')
            torch.save(record, args.run_dir/f'checkpoint-step{step+1}.pt')
            if averaging is not None:
                torch.save(dict(record, model={name:value.detach().cpu()
                                               for name,value in averaging.items()}),
                           args.run_dir/f'checkpoint-step{step+1}-ema.pt')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    ema_result = None
    if averaging is None:
        validation = score(model,*data['validation'],device,'fp32')
        validation.pop('window_nll_nats')
    else:
        # Raw and averaged weights are compared on the scoring device the board uses, and the
        # checkpoint keeps whichever of the two validates better.
        model.to('cpu')
        validation = score(model,*data['validation'],torch.device('cpu'),'fp32')
        validation.pop('window_nll_nats')
        raw_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaging[name])
        averaged = score(model,*data['validation'],torch.device('cpu'),'fp32')
        averaged.pop('window_nll_nats')
        selected = 'ema' if averaged['bpb'] < validation['bpb'] else 'raw'
        if selected == 'raw':
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(raw_weights[name])
        ema_result = {'decay':args.ema_decay,'selected':selected,'raw_bpb':validation['bpb'],
                      'ema_bpb':averaged['bpb'],'validation_ema':averaged}
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    # Provenance for non-baseline recipes only, so the default run still writes exactly the old record.
    recipe = ({} if args.optimizer == 'adamw' and args.adamw_lr == .001 and not teachers
              and averaging is None and args.mtp_weight == 0 and not args.epoch_shuffle
              and args.weight_decay == .1 and args.schedule == 'cosine' else
              {'recipe': {'optimizer':args.optimizer,'adamw_lr':args.adamw_lr,'muon_lr':args.muon_lr,
                          'muon_momentum':args.muon_momentum,'weight_decay':args.weight_decay}})
    if teachers:
        recipe['recipe']['distill'] = {'T':args.distill_T,'alpha':args.distill_alpha,
                                       'teachers':teacher_provenance}
    if args.mtp_weight > 0:
        recipe['recipe']['mtp_weight'] = args.mtp_weight
        recipe['recipe']['mtp_k'] = int(config.get('mtp_k', 0))
    if args.epoch_shuffle:
        recipe['recipe']['epoch_shuffle'] = True
    if args.schedule != 'cosine':
        recipe['recipe']['schedule'] = args.schedule
        recipe['recipe']['cooldown_frac'] = args.cooldown_frac
    if averaging is not None:
        recipe['recipe']['ema_decay'] = args.ema_decay
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              **recipe,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device),**({} if ema_result is None else {'ema':ema_result})}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
