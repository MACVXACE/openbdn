"""OpenBDN: a simulation SDK for the BDN-04 Biological Data Network.

The package models three layers of a BDN-04 link:

* :mod:`openbdn.channel` -- an electro-quasistatic human body communication
  (EQ-HBC) baseband channel with electromyographic (EMG) interference and an
  NLMS adaptive equaliser.
* :mod:`openbdn.crypto` -- neural key agreement with mutually learning
  Tree Parity Machines.
* :mod:`openbdn.storage` -- a constrained binary <-> DNA transcoder that
  respects GC-content and homopolymer limits.

:class:`BDNPipeline` wires the storage and channel layers into one end-to-end
transmission.

Example
-------
>>> from openbdn import BDNPipeline
>>> report = BDNPipeline(seed=7).transmit("hello", snr_db=5.0)
>>> report.recovered_text
'hello'
"""

from __future__ import annotations

from openbdn.channel.eq_hbc import EQChannelSimulator
from openbdn.crypto.vvtpm import SyncResult, VectorValuedTPM, synchronize
from openbdn.pipeline import BDNPipeline, TransmissionReport
from openbdn.storage.dna_codec import DNADecodeError, DNAStorageCodec

__version__ = "0.1.0"

__all__ = [
    "BDNPipeline",
    "DNADecodeError",
    "DNAStorageCodec",
    "EQChannelSimulator",
    "SyncResult",
    "TransmissionReport",
    "VectorValuedTPM",
    "__version__",
    "synchronize",
]
