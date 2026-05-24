from collections import OrderedDict

import torch
from tqdm import tqdm

from pyiqa.archs import build_network
from pyiqa.losses import build_loss
from pyiqa.utils import get_root_logger
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class PEVQABaselineModel(GeneralIQAModel):
    """PEVQA Baseline model for text-to-video quality assessment.

    This model supports:
    - Video quality regression with MOS loss
    - Metric loss (e.g., PLCC loss) for better correlation
    - Multiple validation datasets with different metric configurations
    """

    def __init__(self, opt):
        super().__init__(opt)

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        # Initialize best model for saving
        self.net_best = build_network(self.opt['network']).to(self.device)

        # Define losses
        if train_opt.get('mos_loss_opt'):
            self.cri_mos = build_loss(train_opt['mos_loss_opt']).to(self.device)
        else:
            self.cri_mos = None

        # Define metric-related loss (e.g., PLCC loss)
        if train_opt.get('metric_loss_opt'):
            self.cri_metric = build_loss(train_opt['metric_loss_opt']).to(self.device)
        else:
            self.cri_metric = None

        # Set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def _freeze_backbone(self):
        """Freeze visual encoder backbone for transfer learning."""
        net = self.get_bare_model(self.net)
        if hasattr(net, 'model'):
            for param in net.model.parameters():
                param.requires_grad = True
        logger = get_root_logger()
        logger.info('Frozen visual encoder (net.model) parameters.')

    def _unfreeze_trainable_components(self):
        """Unfreeze trainable components (visual encoder + quality weights)."""
        net = self.get_bare_model(self.net)

        # Freeze all model parameters first
        for param in net.model.parameters():
            param.requires_grad = False

        # Unfreeze visual encoder
        for name, param in net.model.named_parameters():
            if 'visual' in name:
                param.requires_grad = True

        # Make quality_weights trainable
        if hasattr(net, 'quality_weights'):
            net.quality_weights = torch.nn.Parameter(
                net.quality_weights.data.clone()
            )
            logger = get_root_logger()
            logger.info('Made quality_weights trainable.')

        logger = get_root_logger()
        logger.info('PEVQA trainable components: visual encoder + quality_weights')

    def feed_data(self, data):
        """Feed data to the model.

        Args:
            data: Dictionary containing:
                - video: Video tensor [B, T, C, H, W]
                - mos_label: Ground truth MOS labels [B, 1]
        """
        if 'video' in data:
            self.lq = data['video'].to(self.device)

        if 'mos_label' in data:
            self.gt_mos = data['mos_label'].to(self.device)

    def net_forward(self, net):
        """Forward pass through the network.

        Args:
            net: The network module

        Returns:
            Output dictionary containing quality_score
        """
        return net(self.lq)

    def optimize_parameters(self, current_iter):
        """Optimize model parameters with combined losses.

        Args:
            current_iter: Current training iteration
        """
        self.optimizer.zero_grad()

        # Forward pass
        out_dict = self.net_forward(self.net)
        score = out_dict['quality_score']
        self.output_score = score

        l_total = 0
        loss_dict = OrderedDict()

        # MOS loss (main regression loss)
        if self.cri_mos:
            l_mos = self.cri_mos(score, self.gt_mos.squeeze(1))
            l_total += l_mos
            loss_dict['l_mos'] = l_mos

        # Metric loss (for better correlation metrics)
        if self.cri_metric:
            l_metric = self.cri_metric(score, self.gt_mos.squeeze(1))
            l_total += l_metric
            loss_dict['l_metric'] = l_metric

        # Backward pass
        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # Log training metrics
        if self.opt['val'].get('metrics'):
            pred_score = self.output_score.squeeze(1).cpu().detach().numpy()
            gt_mos = self.gt_mos.squeeze(1).cpu().detach().numpy()
            from pyiqa.metrics import calculate_metric
            for name, opt_ in self.opt['val']['metrics'].items():
                self.log_dict[f'train_metrics/{name}'] = calculate_metric(
                    [pred_score, gt_mos], opt_
                )

    def test(self):
        """Test/predict on a single batch."""
        self.net.eval()
        with torch.no_grad():
            self.output_score = self.net_forward(self.net)
        self.net.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Distributed validation wrapper.

        Args:
            dataloader: Validation dataloader
            current_iter: Current iteration
            tb_logger: TensorBoard logger
            save_img: Whether to save images
        """
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Non-distributed validation.

        Args:
            dataloader: Validation dataloader
            current_iter: Current iteration
            tb_logger: TensorBoard logger
            save_img: Whether to save images
        """
        dataset_name = dataloader.dataset.opt['name']
        dataset_opt = dataloader.dataset.opt
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        # Get max_val_samples from dataset config, fallback to global config
        max_samples = dataset_opt.get(
            'max_val_samples', self.opt['val'].get('max_val_samples', 100)
        )

        # Initialize best metric results for this dataset
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)

        if use_pbar:
            pbar = tqdm(total=min(len(dataloader), max_samples), unit='video')

        pred_score = []
        gt_mos = []

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            score = self.output_score['quality_score']
            pred_score.append(score)
            gt_mos.append(self.gt_mos)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Validation {dataset_name}')

            # Check sample limit
            if len(pred_score) >= max_samples:
                break

        if use_pbar:
            pbar.close()

        # Concatenate all predictions and ground truth
        pred_score = torch.cat(pred_score, dim=0).view(-1).cpu().numpy()
        gt_mos = torch.cat(gt_mos, dim=0).view(-1).cpu().numpy()

        # Calculate metrics
        if with_metrics:
            # Get metrics for current dataset only
            dataset_metrics = self.opt['val']['metrics'].get(dataset_name, {})
            if not dataset_metrics:
                # Fallback: if dataset not in metrics config, skip metrics calculation
                logger = get_root_logger()
                logger.warning(f'No metrics configured for dataset {dataset_name}, skipping.')
                return

            # Reset metric results for this dataset
            self.metric_results = {
                metric: 0 for metric in dataset_metrics.keys()
            }

            # Calculate all metrics for this dataset
            from pyiqa.metrics import calculate_metric
            for name, opt_ in dataset_metrics.items():
                self.metric_results[name] = calculate_metric(
                    [pred_score, gt_mos], opt_
                )

            # Update best metrics and save model
            key_metric = self.opt['val'].get('key_metric')
            if key_metric is not None and key_metric in self.metric_results:
                to_update = self._update_best_metric_result(
                    dataset_name,
                    key_metric,
                    self.metric_results[key_metric],
                    current_iter,
                )

                if to_update:
                    # Save all metrics for this dataset
                    for name in dataset_metrics.keys():
                        self._update_metric_result(
                            dataset_name, name, self.metric_results[name], current_iter
                        )
                    # Save best model
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')
            else:
                # Update each metric separately
                updated = []
                for name, opt_ in dataset_metrics.items():
                    tmp_updated = self._update_best_metric_result(
                        dataset_name, name, self.metric_results[name], current_iter
                    )
                    updated.append(tmp_updated)
                # Save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')

            # Log validation metrics
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """Log validation metric values.

        Args:
            current_iter: Current iteration
            dataset_name: Name of validation dataset
            tb_logger: TensorBoard logger
        """
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t #{metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                if (
                    dataset_name in self.best_metric_results
                    and metric in self.best_metric_results[dataset_name]
                ):
                    log_str += (
                        f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                        f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                    )
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        # Log to tensorboard
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f'val_metrics/{dataset_name}/{metric}', value, current_iter
                )
