"""Hook to freeze/unfreeze backbone during training."""
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Hook to freeze/unfreeze backbone during training.

    This hook is useful for transfer learning scenarios where you want to
    first train only the head with a frozen backbone, then fine-tune the
    entire model.

    Args:
        freeze_epochs (int): Number of epochs to freeze the backbone.
            After this, the backbone will be unfrozen. Default: 15.
    """

    def __init__(self, freeze_epochs=15):
        self.freeze_epochs = freeze_epochs
        self.is_frozen = False
        self.is_unfrozen = False

    def before_train_epoch(self, runner):
        """Freeze backbone at start, unfreeze after specified epochs."""
        model = runner.model

        # Freeze backbone at the beginning
        if runner.epoch == 0 and not self.is_frozen:
            self._freeze_backbone(model)
            runner.logger.info(
                f'Freezing backbone for first {self.freeze_epochs} epochs. '
                f'Only training the head.')
            self.is_frozen = True

        # Unfreeze backbone after specified epochs
        elif runner.epoch == self.freeze_epochs and not self.is_unfrozen:
            self._unfreeze_backbone(model)
            runner.logger.info(
                f'Unfreezing backbone at epoch {self.freeze_epochs}. '
                f'Now training the entire model.')
            self.is_unfrozen = True

    def _freeze_backbone(self, model):
        """Freeze all backbone parameters."""
        if hasattr(model, 'module'):
            model = model.module

        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
            runner_model = model.backbone
            runner_model.eval()

    def _unfreeze_backbone(self, model):
        """Unfreeze all backbone parameters."""
        if hasattr(model, 'module'):
            model = model.module

        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = True
            runner_model = model.backbone
            runner_model.train()