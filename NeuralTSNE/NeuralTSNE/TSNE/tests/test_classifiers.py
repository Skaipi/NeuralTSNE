from unittest.mock import patch

import pytest
import lightning as L
import torch

from NeuralTSNE.TSNE.classifiers import Classifier
from NeuralTSNE.TSNE.tests.common import (
    MyDataset,
    DataLoaderMock,
)

from NeuralTSNE.TSNE.tests.fixtures.parametric_tsne_fixtures import (
    default_parametric_tsne_instance,
)

from NeuralTSNE.TSNE.parametric_tsne import ParametricTSNE
from NeuralTSNE.TSNE.tests.fixtures.classifier_fixtures import (
    default_classifier_instance,
    classifier_instance,
)


@pytest.mark.parametrize(
    "classifier_instance",
    [{"shuffle": False, "optimizer": "adam", "lr": 1e-5}],
    indirect=True,
)
def test_classifier_init(classifier_instance):
    classifier_instance, params, mock_exaggeration_status = classifier_instance

    assert isinstance(classifier_instance, Classifier)
    assert classifier_instance.tsne == params["tsne"]
    assert classifier_instance.batch_size == params["tsne"].batch_size
    assert classifier_instance.model == params["tsne"].model
    assert classifier_instance.loss_fn == params["tsne"].loss_fn
    assert (
        classifier_instance.exaggeration_epochs
        == params["tsne"].early_exaggeration_epochs
    )
    assert (
        classifier_instance.exaggeration_value
        == params["tsne"].early_exaggeration_value
    )
    assert classifier_instance.shuffle == params["shuffle"]
    assert classifier_instance.lr == params["lr"]
    assert classifier_instance.optimizer == params["optimizer"]
    assert mock_exaggeration_status.call_count == 1


@pytest.mark.parametrize(
    "default_classifier_instance",
    [{"early_exaggeration_epochs": 0}, {"early_exaggeration_epochs": 10}],
    indirect=True,
)
def test_reset_exaggeration_status(default_classifier_instance):
    classifier_instance, params = default_classifier_instance
    classifier_instance.reset_exaggeration_status()

    params = params["tsne_params"]
    if params["early_exaggeration_epochs"] == 0:
        assert classifier_instance.has_exaggeration_ended == True
    else:
        assert classifier_instance.has_exaggeration_ended == False


@pytest.mark.parametrize(
    "optimizer, expected_instance",
    [
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_set_optimizer(
    default_classifier_instance,
    optimizer: str,
    expected_instance: torch.optim.Optimizer,
):
    classifier_instance, _ = default_classifier_instance

    returned = classifier_instance._set_optimizer(
        optimizer, {"lr": classifier_instance.lr}
    )
    assert isinstance(returned, expected_instance)
    assert returned.param_groups[0]["lr"] == classifier_instance.lr


@pytest.mark.parametrize("optimizer", ["dummy_optimizer", "adom"])
def test_set_optimizer_invalid(default_classifier_instance, optimizer: str):
    classifier_instance, _ = default_classifier_instance

    with pytest.raises(ValueError):
        classifier_instance._set_optimizer(optimizer, {"lr": classifier_instance.lr})


def test_predict_step(default_classifier_instance):
    classifier_instance, params = default_classifier_instance
    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    for i, batch in enumerate(test_data):
        logits = classifier_instance.predict_step(batch, i)
        assert logits.shape == (
            tsne_instance.batch_size,
            params["default_tsne_params"]["n_components"],
        )


@pytest.mark.parametrize("has_P_multiplied", [True, False])
@pytest.mark.parametrize("has_exaggeration_ended", [True, False])
def test_on_train_epoch_end(
    default_classifier_instance, has_P_multiplied: bool, has_exaggeration_ended: bool
):
    classifier_instance, _ = default_classifier_instance

    if has_P_multiplied:
        classifier_instance.P_multiplied = torch.tensor(torch.nan)
    classifier_instance.has_exaggeration_ended = has_exaggeration_ended

    classifier_instance.on_train_epoch_end()

    if has_P_multiplied:
        assert (
            hasattr(classifier_instance, "P_multiplied") is not has_exaggeration_ended
        )
    else:
        assert hasattr(classifier_instance, "P_multiplied") is False


@pytest.mark.parametrize("has_P", [True, False])
def test_on_train_start(default_classifier_instance, has_P: bool):
    classifier_instance, _ = default_classifier_instance
    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    trainer = L.Trainer(fast_dev_run=True)

    if has_P:
        classifier_instance.P = torch.tensor(torch.nan)

    with (
        patch.object(ParametricTSNE, "_calculate_P") as mocked_calculate_P,
        patch.object(
            Classifier, "training_step", autospec=True
        ) as mocked_training_step,
        patch.object(Classifier, "on_train_epoch_start"),
        patch.object(Classifier, "on_train_epoch_end"),
    ):
        mocked_calculate_P.return_value = torch.tensor(torch.nan)
        mocked_training_step.return_value = None

        trainer.fit(classifier_instance, test_data)

    if not has_P:
        assert mocked_calculate_P.call_count == 1
    else:
        assert mocked_calculate_P.call_count == 0

    assert torch.allclose(
        classifier_instance.P, torch.tensor(torch.nan), equal_nan=True
    )


@pytest.mark.parametrize("epochs", [1, 2, 3])
@pytest.mark.parametrize("has_exaggeration_ended", [True, False])
@pytest.mark.parametrize("exaggeration_epochs", [0, 1])
def test_on_train_epoch_start(
    default_classifier_instance,
    epochs: int,
    has_exaggeration_ended: bool,
    exaggeration_epochs: int,
):
    classifier_instance, params = default_classifier_instance

    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    trainer = L.Trainer(max_epochs=epochs, limit_train_batches=1)

    input_P = torch.ones((num_samples, tsne_instance.batch_size))
    classifier_instance.P = input_P

    classifier_instance.has_exaggeration_ended = has_exaggeration_ended
    classifier_instance.exaggeration_epochs = exaggeration_epochs

    with (
        patch.object(Classifier, "on_train_start"),
        patch.object(
            Classifier, "training_step", autospec=True
        ) as mocked_training_step,
        patch.object(Classifier, "on_train_epoch_end"),
    ):
        mocked_training_step.return_value = None

        trainer.fit(classifier_instance, test_data)

    if has_exaggeration_ended and exaggeration_epochs == 0:
        assert torch.allclose(classifier_instance.P_current, input_P)
    elif has_exaggeration_ended:
        assert torch.allclose(
            classifier_instance.P_current,
            input_P * params["default_tsne_params"]["early_exaggeration_value"],
        )

    if (
        not has_exaggeration_ended
        and epochs <= exaggeration_epochs
        and exaggeration_epochs > 0
    ):
        assert torch.allclose(
            classifier_instance.P_current,
            input_P * params["default_tsne_params"]["early_exaggeration_value"],
        )
    elif not has_exaggeration_ended:
        assert torch.allclose(classifier_instance.P_current, input_P)
        assert classifier_instance.has_exaggeration_ended is True


def test_training_step(default_classifier_instance):
    classifier_instance, params = default_classifier_instance

    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    trainer = L.Trainer(fast_dev_run=True, accelerator="cpu")

    input_P = torch.ones((num_samples, tsne_instance.batch_size))
    classifier_instance.P = input_P

    with patch.object(Classifier, "on_train_start"):
        trainer.fit(classifier_instance, test_data)


@pytest.mark.parametrize(
    "optimizer, expected_instance",
    [
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_configure_optimizers(
    default_classifier_instance,
    optimizer: str,
    expected_instance: torch.optim.Optimizer,
):
    classifier_instance, _ = default_classifier_instance
    classifier_instance.optimizer = optimizer

    returned = classifier_instance.configure_optimizers()
    assert isinstance(returned, expected_instance)
    assert returned.param_groups[0]["lr"] == classifier_instance.lr


@pytest.mark.parametrize("optimizer", ["dummy_optimizer", "adom"])
def test_configure_optimizers_invalid(default_classifier_instance, optimizer: str):
    classifier_instance, _ = default_classifier_instance
    classifier_instance.optimizer = optimizer

    with pytest.raises(ValueError):
        classifier_instance.configure_optimizers()
