import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import inspect
from torch.autograd import Variable
from copsolver.analytical_solver import AnalyticalSolver
from commondescentvector.multi_objective_cdv import MultiObjectiveCDV
from trainer.pcgrad import pc_grad_update
from copsolver.peltr import PELTR
from utils.pareto_manager import ParetoManager


class MAMOTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, trainable_params,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.losses_num = len(self.criterion)
        self.max_empirical_losses = self._compute_max_expirical_losses()
        copsolver = AnalyticalSolver()
        self.common_descent_vector = MultiObjectiveCDV(
            copsolver=copsolver, max_empirical_losses=self.max_empirical_losses,
            normalized=True)
        self.trainable_params = trainable_params
        self.opt_losses = self.config['opt_losses']

        self.train_metrics = MetricTracker('a1', 'a2', 'loss', 'weighted_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'weighted_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.pe_ltr = PELTR()
        self.pareto_manager = ParetoManager()

    def _compute_max_expirical_losses(self):
        max_losses = [0] * self.losses_num
        cnt = 0
        
        for batch_idx, (data, target, price) in enumerate(self.data_loader):
            data, target, price = data.to(self.device), target.to(self.device), price.to(self.device)
            cnt += 1

            output = self.model(data)

            for i in range(self.losses_num):
                l = self._cal_loss(self.criterion[i], output, target, price)
                max_losses[i] = (cnt - 1) / cnt * \
                    max_losses[i] + 1 / cnt * l.item()

        return max_losses

    def _cal_loss(self, c, output, target, price):
        para_nums = len(inspect.getargspec(c)[0])
        if para_nums == 2:
            return c(output, target.float())
        elif para_nums == 3:
            return c(output, target.float(), price)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        average_alpha = [0] * self.losses_num
        cnt = 0
        for batch_idx, (data, target, price) in enumerate(self.data_loader):
            cnt += 1
            data, target, price = Variable(data).to(self.device), Variable(target).to(self.device), Variable(price).to(self.device)

            losses_computed = []
            if self.opt_losses == 0 or self.opt_losses == 1:
                output = self.model(data)
                for loss in self.criterion:
                    losses_computed.append(self._cal_loss(loss, output, target, price))

                self.optimizer.zero_grad()
                losses_computed[self.opt_losses].backward()
                self.optimizer.step()
            elif self.opt_losses == 2:
                # calculate the gradients
                gradients = []
                for i, loss in enumerate(self.criterion):
                    # forward pass
                    output = self.model(data)
                    # calculate loss
                    L = self._cal_loss(loss, output, target, price)
                    # zero gradient
                    self.optimizer.zero_grad()
                    # backward pass
                    L.backward()
                    # get gradient for correctness objective
                    gradients.append(self.optimizer.get_gradient())

                # calculate the losses
                # forward pass
                output = self.model(data)

                for i, loss in enumerate(self.criterion):
                    L = self._cal_loss(loss, output, target, price)
                    losses_computed.append(L)

                # get the final loss to compute the common descent vector
                final_loss, alphas = self.common_descent_vector.get_descent_vector(
                    losses_computed, gradients)
                
                self.train_metrics.update('a1', alphas[0])
                self.train_metrics.update('a2', alphas[1])
                # zero gradient
                self.optimizer.zero_grad()
                # backward pass
                final_loss.backward()
                # update parameters
                self.optimizer.step()
            elif self.opt_losses == 3:
                losses_computed = []
                # calculate the gradients
                gradients = []
                self.optimizer.zero_grad()
                for i, loss in enumerate(self.criterion):
                    # forward pass
                    output = self.model(data)
                    # calculate loss
                    L = self._cal_loss(loss, output, target, price)
                    # backward pass
                    L.backward()
                    losses_computed.append(L)
                    # get gradient for correctness objective
                    gradients.append(self.optimizer.get_gradient())
                pc_grad_update(gradients)
                self.optimizer.step()
            elif self.opt_losses == 4:
                # calculate the gradients
                gradients = []
                for i, loss in enumerate(self.criterion):
                    # forward pass
                    output = self.model(data)
                    # calculate loss
                    L = self._cal_loss(loss, output, target, price)
                    # zero gradient
                    self.optimizer.zero_grad()
                    # backward pass
                    L.backward()
                    # get gradient for correctness objective
                    gradients.append(self.optimizer.get_gradient())

                # get the final loss to compute the common descent vector
                a1, a2 = self.pe_ltr.solve(gradients[0], gradients[1])

                # calculate the losses
                # forward pass
                output = self.model(data)

                for i, loss in enumerate(self.criterion):
                    L = self._cal_loss(loss, output, target, price)
                    losses_computed.append(L)

                final_loss = losses_computed[0] * a1 + losses_computed[1] * a2
                
                self.train_metrics.update('a1', a1)
                self.train_metrics.update('a2', a2)
                # zero gradient
                self.optimizer.zero_grad()
                # backward pass
                final_loss.backward()
                # update parameters
                self.optimizer.step()
            elif self.opt_losses == 5:
                output = self.model(data)
                for loss in self.criterion:
                    losses_computed.append(self._cal_loss(loss, output, target, price))
                
                final_loss = losses_computed[0] * self.config['a1'] + losses_computed[1] * self.config['a2']

                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', losses_computed[0].item())
            self.train_metrics.update('weighted_loss', losses_computed[1].item())
            for met in self.metric_ftns:
                para_nums = len(inspect.getargspec(met)[0])
                if para_nums == 2:
                    self.train_metrics.update(met.__name__, met(output, target))
                elif para_nums == 3:
                    self.train_metrics.update(met.__name__, met(output, target, price))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    losses_computed[0].item()))

            if batch_idx == self.len_epoch:
                break
        
        if self.opt_losses == 0:
            print("Optimize only logloss")
        elif self.opt_losses == 1:
            print("Optimize only weighted logloss")
        elif self.opt_losses == 2:
            print("Optimize both logloss and weighted logloss")
            print(average_alpha)
        elif self.opt_losses == 3:
            print("PCGrad")
        elif self.opt_losses == 4:
            print("PELTR")
        else:
            print(self.config['a1'], self.config['a2'])
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, price) in enumerate(self.valid_data_loader):
                data, target, price = data.to(self.device), target.to(self.device), price.to(self.device)

                output = self.model(data)
                loss = self.criterion[0](output, target.float())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                w_loss = self.criterion[1](output, target.float(), price)
                self.valid_metrics.update('weighted_loss', w_loss.item())
                ms = [1 - loss.item(), 1 - w_loss.item()]
                for met in self.metric_ftns:
                    para_nums = len(inspect.getargspec(met)[0])
                    if para_nums == 2:
                        self.valid_metrics.update(met.__name__, met(output, target))
                    elif para_nums == 3:
                        self.valid_metrics.update(met.__name__, met(output, target, price))

        l1, l2 = self.valid_metrics.avg('loss'), self.valid_metrics.avg('weighted_loss')
        self.pareto_manager.add_solution([1 - l1, 1 - l2])
        self.logger.info("pareto: " + str([[1 - i[0][0], 1 - i[0][1]] for i in self.pareto_manager._pareto_front]))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)