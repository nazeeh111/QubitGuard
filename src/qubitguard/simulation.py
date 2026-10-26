"""Circuit-level surface memories and syndrome-consistent graph corrections."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import heapq
import numpy as np
import stim
import pymatching

@dataclass(frozen=True)
class Setting:
    distance: int = 3
    noise: float = .007
    profile: str = 'uniform'
    basis: str = 'x'
    def __post_init__(self):
        if self.distance not in (3, 5, 7):
            raise ValueError('Supported distances: 3, 5, 7')
        if not 0 <= self.noise <= .05:
            raise ValueError('Noise must be between 0 and 0.05')
        if self.profile not in ('uniform', 'biased') or self.basis not in ('x', 'z'):
            raise ValueError('Unknown profile or basis')
    @property
    def key(self):
        return f'd{self.distance}_{self.basis}_{self.profile}_p{self.noise:.6f}'


def make_circuit(s: Setting) -> stim.Circuit:
    """All four circuit-level channels are explicit; biased is measurement-heavy.

    'biased' here means channel imbalance, NOT a Pauli-Z bias. The gate rate
    stays p; measurement/reset rates become 2p and idle-data rate becomes p/4.
    """
    p = s.noise
    return stim.Circuit.generated(
        f'surface_code:rotated_memory_{s.basis}', distance=s.distance, rounds=s.distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p if s.profile == 'uniform' else p/4,
        before_measure_flip_probability=p if s.profile == 'uniform' else 2*p,
        after_reset_flip_probability=p if s.profile == 'uniform' else 2*p,
    )

class Decoder:
    def __init__(self, setting: Setting):
        self.setting = setting
        self.circuit = make_circuit(setting)
        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(self.dem, enable_correlations=True)
        coords = self.circuit.get_detector_coordinates()
        self.coords = np.array([coords[k] for k in range(self.circuit.num_detectors)])
        self.indices = np.rint(self.coords[:, [2, 1, 0]] / [1, 2, 2]).astype(int)
        self.grid_shape = (setting.distance+1,)*3
        self.edges = self.matching.edges()
        self._loop = None
    def sample(self, shots: int, seed: int):
        return self.circuit.compile_detector_sampler(seed=int(seed)).sample(shots, separate_observables=True)
    def predict(self, dets, correlated=False):
        return self.matching.decode_batch(np.ascontiguousarray(dets, dtype=np.uint8), enable_correlations=correlated)[:, 0].astype(np.uint8)
    def grid(self, dets):
        g = np.zeros((len(dets), 2, *self.grid_shape), np.float32)
        t,y,x = self.indices.T
        g[:,0,t,y,x] = dets
        g[:,1,t,y,x] = 1
        return g
    def correction(self, syndrome, correlated=False):
        e = self.matching.decode_to_edges_array(syndrome, enable_correlations=correlated)
        chosen = set()
        lookup = {tuple(sorted((a, -1 if b is None else b))):i for i,(a,b,_) in enumerate(self.edges)}
        for a,b in e:
            chosen.symmetric_difference_update([lookup[tuple(sorted((int(a),int(b))))]])
        return chosen
    def certificate(self, syndrome, chosen, logical=None):
        parity = np.zeros(self.circuit.num_detectors, np.uint8)
        obs = 0
        for i in chosen:
            a,b,data = self.edges[i]
            parity[a] ^= 1
            if b is not None: parity[b] ^= 1
            obs ^= int(0 in data['fault_ids'])
        return bool(np.array_equal(parity, np.asarray(syndrome,dtype=np.uint8)) and (logical is None or obs == logical))
    def logical_loop(self):
        """Shortest odd-logical cycle in the doubled matching graph.

        Boundary endpoints are merged as in PyMatching; boundary parity is free.
        This is a graphlike DEM representative, not a physical-qubit chain.
        """
        if self._loop is not None: return self._loop
        adj = {}
        for i,(a,b,data) in enumerate(self.edges):
            b = -1 if b is None else b
            flip = int(0 in data['fault_ids'])
            w = max(float(data['weight']), 1e-9)
            adj.setdefault(a,[]).append((b,flip,w,i))
            adj.setdefault(b,[]).append((a,flip,w,i))
        best = (float('inf'), None)
        for origin in adj:
            start = (origin,0); target=(origin,1)
            dist={start:0.}; prev={}; queue=[(0.,start)]
            while queue:
                cost,u=heapq.heappop(queue)
                if cost != dist[u]: continue
                if cost >= best[0]: break
                if u == target:
                    path=set()
                    while u != start:
                        u0,ei=prev[u]; path.symmetric_difference_update([ei]); u=u0
                    best=(cost,path); break
                for b,flip,w,ei in adj[u[0]]:
                    v=(b,u[1]^flip); nc=cost+w
                    if nc < dist.get(v,float('inf')):
                        dist[v]=nc; prev[v]=(u,ei); heapq.heappush(queue,(nc,v))
        if best[1] is None: raise RuntimeError('No logical graph cycle found')
        self._loop=best[1]
        if not self.certificate(np.zeros(self.circuit.num_detectors,np.uint8), self._loop, 1):
            raise RuntimeError('Invalid logical loop')
        return self._loop
    def edge_json(self, chosen):
        return [dict(a=int(self.edges[i][0]),b=None if self.edges[i][1] is None else int(self.edges[i][1]),logical=int(0 in self.edges[i][2]['fault_ids'])) for i in sorted(chosen)]
    @property
    def circuit_hash(self):
        return hashlib.sha256(str(self.circuit).encode()).hexdigest()


def context_features(dets, standard, correlated, noise):
    """Features available at inference only; never true observable labels."""
    return np.stack([standard, correlated, standard != correlated,
                     np.mean(dets,axis=1),np.full(len(dets),noise/.01)],axis=1).astype(np.float32)
