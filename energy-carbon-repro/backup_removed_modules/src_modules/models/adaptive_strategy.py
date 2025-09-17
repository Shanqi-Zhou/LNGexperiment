# Import core model components for paper reproduction
from .mlr import MLR
from .gpr import ExactGPR
from .residual_framework import ResidualModelingPipeline
try:
    from .cross_modal_fusion import CrossModalFusionWithResidual
    CROSS_MODAL_AVAILABLE = True
except ImportError:
    CROSS_MODAL_AVAILABLE = False
    print("Warning: CrossModalFusion not available, using fallback model")

# Define thresholds based on paper requirements
SMALL_SAMPLE_THRESHOLD = 20000
MEDIUM_SAMPLE_THRESHOLD = 100000

def create_adaptive_model(n_samples, dynamic_dim=9, static_dim=32, seq_len=180):
    """
    Factory function for creating the paper-specified model architecture.

    For LNG paper reproduction, we use the cross-modal fusion architecture
    with MLR+GPR residual modeling as specified in the paper.

    Args:
        n_samples (int): Number of training samples
        dynamic_dim (int): Dynamic feature dimension (9 categories)
        static_dim (int): Static feature dimension (32)
        seq_len (int): Sequence length (180 for 30min windows)

    Returns:
        Model instance for paper reproduction
    """
    print(f"Creating model for n_samples = {n_samples}...")

    # For paper reproduction, prioritize cross-modal fusion architecture
    if CROSS_MODAL_AVAILABLE and n_samples > MEDIUM_SAMPLE_THRESHOLD:
        print("Strategy: Paper reproduction -> CrossModalFusion + MLR/GPR Residual")
        model = CrossModalFusionWithResidual(
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            seq_len=seq_len,
            d_model=128,
            n_heads=4,
            n_layers=3
        )
        return model
    else:
        # Fallback to simpler MLR+GPR for smaller datasets or if cross-modal not available
        print("Strategy: MLR + ExactGPR Residual Pipeline")
        base_model = MLR()
        residual_model = ExactGPR()
        return ResidualModelingPipeline(base_model=base_model, residual_model=residual_model)

# Simplified test
if __name__ == '__main__':
    print("--- Paper Reproduction Model Strategy ---")

    # Test model creation
    model = create_adaptive_model(150000)
    print(f"Model Type: {type(model).__name__}")
    print("-" * 40)