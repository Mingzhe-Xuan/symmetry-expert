import torch

# e3nn 0.4.4 ships immutable Clebsch-Gordan constants containing Python slice
# objects. PyTorch >=2.6 defaults torch.load to weights-only mode, so explicitly
# allow only that required built-in before any e3nn module is imported.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from .__version__ import __version__
