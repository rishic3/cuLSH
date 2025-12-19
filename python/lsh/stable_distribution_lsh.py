class StableDistributionLSH:

    def __init__(self, n_projections, bucket_length):
        self._n_projections = n_projections
        self._bucket_length = bucket_length

    @property
    def bucket_length(self):
        return self._bucket_length

    @property
    def n_projections(self):
        return self._n_projections
