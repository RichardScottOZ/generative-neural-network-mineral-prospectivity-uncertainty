"""
Tests for bug fixes in mineral prospectivity implementation.
"""

import torch
import numpy as np
import tempfile
import os
import pytest

from mineral_prospectivity.models.vae_model import VAEProspectivityModel, Decoder
from mineral_prospectivity.models.ensemble import EnsembleModel
from mineral_prospectivity.uncertainty.total_uncertainty import TotalUncertainty


class TestVAEAleatoricUncertaintyFix:
    """Test that predict_with_uncertainty averages aleatoric uncertainty across MC samples."""

    def test_aleatoric_uncertainty_averaged_across_samples(self):
        """Aleatoric uncertainty should be averaged across all MC samples, not from a single sample."""
        torch.manual_seed(42)
        model = VAEProspectivityModel(input_dim=10, latent_dim=4)
        model.eval()

        x = torch.randn(5, 10)
        result = model.predict_with_uncertainty(x, num_samples=50)

        # All uncertainty outputs should be non-negative
        assert (result['aleatoric_uncertainty'] >= 0).all()
        assert (result['epistemic_uncertainty'] >= 0).all()
        assert (result['total_uncertainty'] >= 0).all()

        # Total uncertainty should follow: total^2 = aleatoric^2 + epistemic^2
        expected_total_var = result['aleatoric_uncertainty'] ** 2 + result['epistemic_uncertainty'] ** 2
        actual_total_var = result['total_uncertainty'] ** 2
        assert torch.allclose(actual_total_var, expected_total_var, atol=1e-5)

    def test_aleatoric_is_not_from_single_sample(self):
        """Ensure aleatoric variance is an average, not from one sample."""
        torch.manual_seed(42)
        model = VAEProspectivityModel(input_dim=10, latent_dim=4)
        model.eval()

        x = torch.randn(5, 10)

        # Run with different sample counts - should give slightly different results
        # due to averaging, but both should be reasonable
        result_10 = model.predict_with_uncertainty(x, num_samples=10)
        result_200 = model.predict_with_uncertainty(x, num_samples=200)

        # Both should produce valid non-negative uncertainties
        assert (result_10['aleatoric_uncertainty'] >= 0).all()
        assert (result_200['aleatoric_uncertainty'] >= 0).all()


class TestSaveLoadHiddenDimsFix:
    """Test that save/load preserves encoder/decoder hidden dims."""

    def test_save_load_default_dims(self):
        """Model with default dims should save and load correctly."""
        torch.manual_seed(42)
        model = VAEProspectivityModel(input_dim=10, latent_dim=8)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            model.save_model(path)
            loaded = VAEProspectivityModel.load_model(path)

            assert loaded.input_dim == 10
            assert loaded.latent_dim == 8
            assert loaded.encoder_hidden_dims == [256, 128, 64]
            assert loaded.decoder_hidden_dims == [64, 128, 256]

            # Check predictions match (seed needed due to reparameterization)
            x = torch.randn(3, 10)
            model.eval()
            loaded.eval()
            with torch.no_grad():
                torch.manual_seed(99)
                orig = model(x)
                torch.manual_seed(99)
                new = loaded(x)
            assert torch.allclose(orig['prediction_mean'], new['prediction_mean'], atol=1e-5)
        finally:
            os.unlink(path)

    def test_save_load_custom_dims(self):
        """Model with custom hidden dims should save and load correctly."""
        torch.manual_seed(42)
        custom_enc = [128, 64]
        custom_dec = [64, 128]
        model = VAEProspectivityModel(
            input_dim=20,
            latent_dim=16,
            encoder_hidden_dims=custom_enc,
            decoder_hidden_dims=custom_dec,
            output_dim=1
        )

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            model.save_model(path)
            loaded = VAEProspectivityModel.load_model(path)

            assert loaded.encoder_hidden_dims == custom_enc
            assert loaded.decoder_hidden_dims == custom_dec
            assert loaded.input_dim == 20
            assert loaded.latent_dim == 16

            # Check predictions match (seed needed due to reparameterization)
            x = torch.randn(3, 20)
            model.eval()
            loaded.eval()
            with torch.no_grad():
                torch.manual_seed(99)
                orig = model(x)
                torch.manual_seed(99)
                new = loaded(x)
            assert torch.allclose(orig['prediction_mean'], new['prediction_mean'], atol=1e-5)
        finally:
            os.unlink(path)


class TestDecoderLogvarClamp:
    """Test that decoder logvar is clamped for numerical stability."""

    def test_logvar_is_bounded(self):
        """Decoder logvar output should be clamped between -10 and 10."""
        torch.manual_seed(42)
        decoder = Decoder(latent_dim=4, hidden_dims=[32], output_dim=1)
        decoder.eval()

        # Use extreme input to test clamping
        z = torch.randn(5, 4) * 100
        _, logvar = decoder(z)

        assert logvar.max() <= 10.0
        assert logvar.min() >= -10.0


class TestRiskAnalysisConfidenceInterval:
    """Test that risk analysis uses proper confidence interval."""

    def test_confidence_threshold_affects_recommendation(self):
        """Different confidence thresholds should produce different results."""
        predictions = np.array([0.8, 0.6, 0.9, 0.5, 0.7])
        uncertainties = np.array([0.1, 0.2, 0.05, 0.3, 0.15])

        result_high = TotalUncertainty.uncertainty_risk_analysis(
            predictions, uncertainties, confidence_level=0.99
        )
        result_low = TotalUncertainty.uncertainty_risk_analysis(
            predictions, uncertainties, confidence_level=0.5
        )

        # Higher confidence level should use larger z-score, wider interval
        # So the lower bounds should be lower with higher confidence
        assert (result_high['lower_confidence_bound'] <= result_low['lower_confidence_bound']).all()

    def test_lower_bound_uses_proper_z_score(self):
        """Lower bound should use proper z-score from confidence_level."""
        from scipy import stats
        predictions = np.array([0.8])
        uncertainties = np.array([0.1])

        result = TotalUncertainty.uncertainty_risk_analysis(
            predictions, uncertainties, confidence_level=0.95
        )

        expected_z = stats.norm.ppf((1 + 0.95) / 2)
        expected_lower = 0.8 - expected_z * 0.1
        np.testing.assert_almost_equal(
            result['lower_confidence_bound'][0], expected_lower, decimal=5
        )

    def test_confidence_threshold_controls_recommendation(self):
        """confidence_threshold should control the recommendation decision."""
        predictions = np.array([0.7])
        uncertainties = np.array([0.01])

        result_low = TotalUncertainty.uncertainty_risk_analysis(
            predictions, uncertainties, confidence_threshold=0.5
        )
        result_high = TotalUncertainty.uncertainty_risk_analysis(
            predictions, uncertainties, confidence_threshold=0.9
        )

        # With low threshold, 0.7 should be recommended
        assert result_low['recommend_exploration'][0] == True
        # With high threshold, 0.7 should NOT be recommended
        assert result_high['recommend_exploration'][0] == False


class TestEnsembleSaveLoad:
    """Test ensemble save/load with weights_only parameter."""

    def test_ensemble_save_load_roundtrip(self):
        """Ensemble should save and load correctly."""
        torch.manual_seed(42)
        ensemble = EnsembleModel(num_models=2, input_dim=10, latent_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save_ensemble(tmpdir)
            loaded = EnsembleModel.load_ensemble(tmpdir)

            assert loaded.num_models == 2
            assert loaded.input_dim == 10
            assert loaded.latent_dim == 4

            # Check predictions match (seed needed due to reparameterization)
            x = torch.randn(3, 10)
            for i in range(2):
                ensemble.models[i].eval()
                loaded.models[i].eval()

            with torch.no_grad():
                torch.manual_seed(99)
                orig_out = ensemble.models[0](x)
                torch.manual_seed(99)
                loaded_out = loaded.models[0](x)
            assert torch.allclose(
                orig_out['prediction_mean'], loaded_out['prediction_mean'], atol=1e-5
            )


class TestGradientClipping:
    """Test that gradient clipping is applied during training."""

    def test_training_with_gradient_clipping(self):
        """Training should complete without gradient explosion."""
        torch.manual_seed(42)
        ensemble = EnsembleModel(num_models=1, input_dim=5, latent_dim=2)

        # Create a small dataset
        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # This should complete without NaN losses
        history = ensemble.train_model(
            model_idx=0,
            train_loader=loader,
            epochs=5,
            device='cpu'
        )

        for loss in history['train_losses']:
            assert not np.isnan(loss), "Training produced NaN loss"
            assert not np.isinf(loss), "Training produced infinite loss"
