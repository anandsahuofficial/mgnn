"""Feature engineering and graph construction modules."""

from mgnn.features.graph_constructor import CrystalGraphConstructor
from mgnn.features.atom_features import AtomFeaturizer
from mgnn.features.composition_features import CompositionFeaturizer

__all__ = [
    'CrystalGraphConstructor',
    'AtomFeaturizer',
    'CompositionFeaturizer'
]


# # Test individual modules
# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn

# # Test atom featurizer
# uv run python -m mgnn.features.atom_features

# # Test graph constructor
# uv run python -m mgnn.features.graph_constructor

# # Test composition featurizer
# uv run python -m mgnn.features.composition_features