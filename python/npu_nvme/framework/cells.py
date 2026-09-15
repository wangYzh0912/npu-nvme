"""Scalar checked training with an explicit optimizer dependency."""
import mindspore as ms
from mindspore import ops, nn, Tensor


class _CheckedScalarLossCell(nn.Cell):
    """Make the causal-LM training contract explicit inside the graph."""

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network
        self.check_numerics = ops.CheckNumerics()

    def construct(self, *inputs):
        loss = self.network(*inputs)
        # These Python checks execute while GRAPH_MODE specialises the graph,
        # before gradients or an optimizer update can be emitted.
        if isinstance(loss, (tuple, list)):
            raise TypeError(
                "training network returned an inference tuple; call "
                "network.set_train(True) and return one scalar loss")
        if not isinstance(loss, Tensor):
            raise TypeError("training network must return a Tensor loss")
        if loss.size != 1:
            raise ValueError(
                f"training loss must contain one value, got shape {loss.shape}")
        if loss.ndim != 0:
            loss = ops.reshape(loss, ())
        # CheckNumerics is an identity for finite input and raises before the
        # backward/optimizer dependency for NaN or Inf.
        return self.check_numerics(loss)


class TrainOneStepCell(nn.Cell):
    @property
    def checkpoint_loss_scale(self):
        # value_and_grad below computes unscaled gradients.
        return 1.0

    def __init__(self, network, optimizer):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_train(True)
        self.network.set_grad()
        self.loss_network = _CheckedScalarLossCell(self.network)
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(
            self.loss_network, grad_position=None,
            weights=self.optimizer.parameters)

        self.depend = ops.Depend()

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        return self.depend(loss, self.optimizer(grads))
