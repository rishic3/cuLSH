"""
Locality Sensitive Hashing on GPUs
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = [
    "Candidates",
    "MinHashCore",
    "MinHashIndex",
    "PSLSHCore",
    "PSLSHIndex",
    "RPLSHCore",
    "RPLSHIndex",
]

class Candidates:
    def empty(self) -> bool: ...
    def get_counts(self, as_cupy: bool = False) -> typing.Any: ...
    def get_indices(self, as_cupy: bool = False) -> typing.Any: ...
    def get_offsets(self, as_cupy: bool = False) -> typing.Any: ...
    @property
    def n_queries(self) -> int: ...
    @property
    def n_total_candidates(self) -> int: ...

class MinHashCore:
    def __init__(
        self,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        seed: typing.SupportsInt = 42,
    ) -> None: ...
    def fit(
        self,
        arg0: typing.Any,
        arg1: typing.Any,
        arg2: typing.SupportsInt,
        arg3: typing.SupportsInt,
    ) -> MinHashIndex: ...
    def fit_query(
        self,
        arg0: typing.Any,
        arg1: typing.Any,
        arg2: typing.SupportsInt,
        arg3: typing.SupportsInt,
    ) -> Candidates: ...
    def query(
        self,
        indices: typing.Any,
        indptr: typing.Any,
        n_queries: typing.SupportsInt,
        index: MinHashIndex,
    ) -> Candidates: ...
    @property
    def n_hash_tables(self) -> int: ...
    @property
    def n_hashes(self) -> int: ...
    @property
    def seed(self) -> int: ...

class MinHashIndex:
    @staticmethod
    def load(
        candidate_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        bucket_signatures: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8],
        bucket_candidate_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        table_bucket_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        hash_a: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32],
        hash_b: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32],
        n_total_candidates: typing.SupportsInt,
        n_total_buckets: typing.SupportsInt,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        sig_nbytes: typing.SupportsInt,
        n_features: typing.SupportsInt,
        seed: typing.SupportsInt,
    ) -> MinHashIndex: ...
    def empty(self) -> bool: ...
    def get_bucket_candidate_offsets(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def get_bucket_signatures(self) -> numpy.typing.NDArray[numpy.uint8]: ...
    def get_candidate_indices(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def get_hash_a(self) -> numpy.typing.NDArray[numpy.uint32]: ...
    def get_hash_b(self) -> numpy.typing.NDArray[numpy.uint32]: ...
    def get_table_bucket_offsets(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def size_bytes(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_hash_tables(self) -> int: ...
    @property
    def n_hashes(self) -> int: ...
    @property
    def n_total_buckets(self) -> int: ...
    @property
    def n_total_candidates(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def sig_nbytes(self) -> int: ...

class PSLSHCore:
    def __init__(
        self,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        window_size: typing.SupportsInt,
        seed: typing.SupportsInt = 42,
    ) -> None: ...
    def fit_double(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> PSLSHIndex: ...
    def fit_float(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> PSLSHIndex: ...
    def fit_query_double(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> Candidates: ...
    def fit_query_float(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> Candidates: ...
    def query_double(
        self,
        Q: typing.Any,
        n_queries: typing.SupportsInt,
        index: PSLSHIndex,
        batch_size: typing.SupportsInt | None = None,
    ) -> Candidates: ...
    def query_float(
        self,
        Q: typing.Any,
        n_queries: typing.SupportsInt,
        index: PSLSHIndex,
        batch_size: typing.SupportsInt | None = None,
    ) -> Candidates: ...
    @property
    def n_hash_tables(self) -> int: ...
    @property
    def n_hashes(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def window_size(self) -> int: ...

class PSLSHIndex:
    @staticmethod
    def load_double(
        candidate_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        bucket_signatures: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8],
        bucket_candidate_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        table_bucket_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        projection: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        bias: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        n_total_candidates: typing.SupportsInt,
        n_total_buckets: typing.SupportsInt,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        sig_nbytes: typing.SupportsInt,
        n_features: typing.SupportsInt,
        seed: typing.SupportsInt,
    ) -> PSLSHIndex: ...
    @staticmethod
    def load_float(
        candidate_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        bucket_signatures: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8],
        bucket_candidate_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        table_bucket_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        projection: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        bias: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        n_total_candidates: typing.SupportsInt,
        n_total_buckets: typing.SupportsInt,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        sig_nbytes: typing.SupportsInt,
        n_features: typing.SupportsInt,
        seed: typing.SupportsInt,
    ) -> PSLSHIndex: ...
    def empty(self) -> bool: ...
    def get_bias(self) -> typing.Any: ...
    def get_bucket_candidate_offsets(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def get_bucket_signatures(self) -> numpy.typing.NDArray[numpy.uint8]: ...
    def get_candidate_indices(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def get_projection_matrix(self) -> typing.Any: ...
    def get_table_bucket_offsets(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def size_bytes(self) -> int: ...
    @property
    def is_double(self) -> bool: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_hash_tables(self) -> int: ...
    @property
    def n_hashes(self) -> int: ...
    @property
    def n_total_buckets(self) -> int: ...
    @property
    def n_total_candidates(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def sig_nbytes(self) -> int: ...

class RPLSHCore:
    def __init__(
        self,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        seed: typing.SupportsInt = 42,
    ) -> None: ...
    def fit_double(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> RPLSHIndex: ...
    def fit_float(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> RPLSHIndex: ...
    def fit_query_double(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> Candidates: ...
    def fit_query_float(
        self, arg0: typing.Any, arg1: typing.SupportsInt, arg2: typing.SupportsInt
    ) -> Candidates: ...
    def query_double(
        self,
        Q: typing.Any,
        n_queries: typing.SupportsInt,
        index: RPLSHIndex,
        batch_size: typing.SupportsInt | None = None,
    ) -> Candidates: ...
    def query_float(
        self,
        Q: typing.Any,
        n_queries: typing.SupportsInt,
        index: RPLSHIndex,
        batch_size: typing.SupportsInt | None = None,
    ) -> Candidates: ...
    @property
    def n_hash_tables(self) -> int: ...
    @property
    def n_hashes(self) -> int: ...
    @property
    def seed(self) -> int: ...

class RPLSHIndex:
    @staticmethod
    def load_double(
        candidate_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        bucket_signatures: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8],
        bucket_candidate_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        table_bucket_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        projection: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        n_total_candidates: typing.SupportsInt,
        n_total_buckets: typing.SupportsInt,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        sig_nbytes: typing.SupportsInt,
        n_features: typing.SupportsInt,
        seed: typing.SupportsInt,
    ) -> RPLSHIndex: ...
    @staticmethod
    def load_float(
        candidate_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        bucket_signatures: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8],
        bucket_candidate_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        table_bucket_offsets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        projection: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
        n_total_candidates: typing.SupportsInt,
        n_total_buckets: typing.SupportsInt,
        n_hash_tables: typing.SupportsInt,
        n_hashes: typing.SupportsInt,
        sig_nbytes: typing.SupportsInt,
        n_features: typing.SupportsInt,
        seed: typing.SupportsInt,
    ) -> RPLSHIndex: ...
    def empty(self) -> bool: ...
    def get_bucket_candidate_offsets(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def get_bucket_signatures(self) -> numpy.typing.NDArray[numpy.uint8]: ...
    def get_candidate_indices(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def get_projection_matrix(self) -> typing.Any: ...
    def get_table_bucket_offsets(self) -> numpy.typing.NDArray[numpy.int32]: ...
    def size_bytes(self) -> int: ...
    @property
    def is_double(self) -> bool: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_hash_tables(self) -> int: ...
    @property
    def n_hashes(self) -> int: ...
    @property
    def n_total_buckets(self) -> int: ...
    @property
    def n_total_candidates(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def sig_nbytes(self) -> int: ...
