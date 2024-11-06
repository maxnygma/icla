import math
import scipy.optimize
import torch


class TemperatureWrapper:
    def __init__(self, net):
        self.net = net
        self.device = next(iter(self.net.parameters())).device
        self.temperature = None

    def fit_temperature(self, data_loader):
        self.net.eval()
        logits = []
        targets = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].to(self.device)
                logit = self.net(data)
                logits.append(logit)
                targets.append(batch['label'].to(self.device))
        logits, targets = torch.cat(logits), torch.cat(targets)
        self.temperature = self._tune_temperature(logits, targets)

    def forward(self, x, **kwargs):
        if self.temperature is None:
            raise RuntimeError("Fit temperature first.")
        result = self.net(x, **kwargs)
        if kwargs.get("return_ensemble", False):
            result = list(result)
            result[0] = result[0] / self.temperature
            result[1] = result[1] / self.temperature
        else:
            result = result / self.temperature
        return result

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def __getattr__(self, name):
        return getattr(self.net, name)

    @staticmethod
    def _tune_temperature(logits, targets, tol=1e-3, min_t=0.001, max_t=100):
        print('\nTune temperature...', flush=True)
        def compute_loss(log_t):
            return torch.nn.functional.cross_entropy(logits / math.exp(log_t), targets).item()
        with torch.no_grad():
            print(f"\nNLL without temperature: {compute_loss(0)}")
            # Find upper bound.
            log_t_low = math.log(min_t)
            log_t_high = log_t_low + 1
            loss = compute_loss(log_t_high)
            while log_t_high < math.log(max_t):
                new_loss = compute_loss(log_t_high + 1)
                if new_loss < loss:
                    log_t_low = log_t_high
                    log_t_high += 1
                    loss = new_loss
                else:
                    break
            log_t_high = min(math.log(max_t), log_t_high + 1)
            # Find minimum.
            log_t = scipy.optimize.golden(compute_loss, brack=(log_t_low, log_t_high), tol=tol)
            final_loss = compute_loss(log_t)
        t = math.exp(log_t)
        print(f'\nFound temperature {t} in bounds ({math.exp(log_t_low)}, {math.exp(log_t_high)}) with NLL {final_loss}', flush=True)
        return t
