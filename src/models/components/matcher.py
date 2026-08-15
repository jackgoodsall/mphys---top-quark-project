
"""
Code sourced from https://github.com/samvanstroud/hepattn/blob/main/src/hepattn/models/matcher.py
"""


import atexit
import contextlib
import time
import warnings
from multiprocessing import get_context, shared_memory
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Literal

import numpy as np
import scipy
import torch
from torch import nn

_POOL_LOCK = Lock()
_THREAD_POOLS: dict[int, ThreadPool] = {}
_PROCESS_POOLS = {}


def _get_thread_pool(n_jobs: int) -> ThreadPool:
    with _POOL_LOCK:
        pool = _THREAD_POOLS.get(n_jobs)
        if pool is None:
            pool = ThreadPool(processes=n_jobs)
            _THREAD_POOLS[n_jobs] = pool
        return pool


def _get_process_pool(n_jobs: int):
    """Get persistent multiprocessing pool using spawn method."""
    with _POOL_LOCK:
        pool = _PROCESS_POOLS.get(n_jobs)
        if pool is None:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=n_jobs)
            _PROCESS_POOLS[n_jobs] = pool
        return pool


@atexit.register
def _close_pools() -> None:
    """Clean up thread and process pools at exit."""
    for pool in list(_THREAD_POOLS.values()):
        try:
            pool.close()
            pool.join()
        except Exception:  # noqa: BLE001, S110
            pass

    for pool in list(_PROCESS_POOLS.values()):
        try:
            pool.close()
            pool.join(timeout=1.0)
        except Exception:  # noqa: BLE001,
            try:
                pool.terminate()
                pool.join(timeout=1.0)
            except Exception:  # noqa: BLE001, S110
                pass


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


SOLVERS = {
    "scipy": solve_scipy,
}




def match_individual(solver_fn, cost: np.ndarray, default_idx: np.ndarray) -> np.ndarray:
    pred_idx = np.asarray(solver_fn(cost), dtype=np.int32)

    if solver_fn is SOLVERS["scipy"]:
        remaining = np.ones(default_idx.shape[0], dtype=np.bool_)
        remaining[pred_idx] = False
        pred_idx = np.concatenate([pred_idx, default_idx[remaining]])

    return pred_idx


def match_parallel(solver_fn, costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int, n_jobs: int = 8) -> torch.Tensor:
    """Thread-based parallel matching across batch."""
    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    default_idx = np.arange(pred_dim, dtype=np.int32)

    if n_jobs <= 1 or batch_size <= 1:
        results = [match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx) for i in range(batch_size)]
        return torch.from_numpy(np.stack(results, axis=0))

    def _run(i: int) -> np.ndarray:
        return match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx)

    pool = _get_thread_pool(n_jobs)
    results = pool.map(_run, range(batch_size), chunksize=chunk_size)
    return torch.from_numpy(np.stack(results, axis=0))


def _mp_match_task(args: tuple[str, str, tuple[int, int, int], str, int, int, int]) -> np.ndarray:
    solver_name, shm_name, shape, dtype_str, i, length, pred_dim = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        costs_t = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        default_idx = np.arange(pred_dim, dtype=np.int32)
        cost = costs_t[i][:length]
        return match_individual(SOLVERS[solver_name], cost, default_idx)
    finally:
        shm.close()


def match_multiprocess(
    solver_name: str,
    costs_t: np.ndarray,
    lengths_np: np.ndarray,
    pred_dim: int,
    n_jobs: int = 8,
) -> torch.Tensor:
    """Multiprocess matching using shared memory to bypass GIL.

    Raises:
        ValueError: If solver_name is not in the available SOLVERS.
    """
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(SOLVERS.keys())}")

    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs

    shm = shared_memory.SharedMemory(create=True, size=costs_t.nbytes)
    try:
        shm_arr = np.ndarray(costs_t.shape, dtype=costs_t.dtype, buffer=shm.buf)
        shm_arr[...] = costs_t

        tasks = [(solver_name, shm.name, costs_t.shape, costs_t.dtype.str, i, int(lengths_np[i]), pred_dim) for i in range(batch_size)]

        pool = _get_process_pool(n_jobs)
        results = pool.map(_mp_match_task, tasks, chunksize=chunk_size)
        return torch.from_numpy(np.stack(results, axis=0))
    finally:
        try:
            shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()


class Matcher(nn.Module):
    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
        parallel_solver: bool = False,
        parallel_backend: Literal["thread", "process"] = "thread",
        n_jobs: int = 8,
        verbose: bool = False,
    ):
        super().__init__()
        """ Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver : str
            The default solving algorithm to use.
        adaptive_solver : bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval : bool
            Interval for checking which solver is the fastest.
        parallel_solver : bool
            If true, then the solver will use a parallel implementation to speed up the matching.
        parallel_backend : str
            Parallel backend when parallel_solver is True. One of: 'thread', 'process'.
        n_jobs: int
            Number of jobs to use for parallel matching. Only used if parallel_solver is True.
        verbose : bool
            If true, extra information on solver timing is printed.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        if parallel_backend not in {"thread", "process"}:
            raise ValueError(f"parallel_backend must be 'thread' or 'process', got: {parallel_backend}")
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs
        self.step = 0
        self.verbose = verbose

    def compute_matching(self, costs, object_valid_mask=None, query_valid_mask=None):
        if object_valid_mask is None:
            object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=torch.bool)

        object_valid_mask = object_valid_mask.detach().bool()
        pred_dim = costs.shape[1]
        valid_np = object_valid_mask.cpu().numpy()
        lengths_np = valid_np.sum(axis=1).astype(np.int32, copy=False)

        # If we have invalid/padded queries, set their costs to a high value
        # so they won't be matched to valid targets
        if query_valid_mask is not None:
            query_valid_mask = query_valid_mask.detach().bool()
            # Set costs for invalid queries to max float32 value
            # costs shape: [batch, num_pred, num_target]
            invalid_query_mask = ~query_valid_mask.unsqueeze(-1)  # [batch, num_pred, 1]
            costs = np.where(invalid_query_mask.cpu().numpy(), np.finfo(np.float32).max / 10, costs)

        # Compact arbitrary valid target columns to the prefix expected by the
        # solver.  The caller already treats matched targets as an exchangeable
        # set, so this preserves the target-order contract while avoiding the
        # old ``[:sum(valid)]`` assumption for masks such as [False, True].
        costs_t_raw = costs.swapaxes(1, 2)
        max_valid = int(lengths_np.max()) if len(lengths_np) else 0
        costs_t = np.zeros((len(costs_t_raw), max_valid, pred_dim), dtype=costs_t_raw.dtype)
        for batch_idx, valid_row in enumerate(valid_np):
            positions = np.flatnonzero(valid_row)
            if positions.size:
                costs_t[batch_idx, :positions.size] = costs_t_raw[batch_idx, positions]

        if self.parallel_solver:
            # [batch, true, pred], with valid targets compacted above.
            if self.parallel_backend == "thread":
                return match_parallel(SOLVERS[self.solver], costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)
            return match_multiprocess(self.solver, costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)

        # Sequential matching
        default_idx = np.arange(pred_dim, dtype=np.int32)
        idxs = []

        for k in range(len(costs)):
            cost = costs_t[k, : lengths_np[k]]
            pred_idx = match_individual(SOLVERS[self.solver], cost, default_idx)
            idxs.append(pred_idx)

        return torch.from_numpy(np.stack(idxs))

    @torch.no_grad()
    def forward(self, costs, object_valid_mask=None, query_valid_mask=None):
        # Convert costs to numpy on CPU for solver compatibility
        costs = costs.detach().to(torch.float32).cpu().numpy()

        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)

        pred_idxs = self.compute_matching(costs, object_valid_mask, query_valid_mask)
        self.step += 1

        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs):
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        for solver in SOLVERS:
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[solver] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {solver}, took {solver_times[solver]:.2f}s")

        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != self.solver:
                print(f"Adaptive LAP Solver: Switching from {self.solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {self.solver} solver\n")

        self.solver = fastest_solver


class BruteForceGPUMatcher(nn.Module):
    """GPU-resident matcher for small T via exhaustive permutation enumeration.

    Replaces scipy Hungarian matching with a fully on-GPU brute-force search
    over all P(Q, T) = Q!/(Q-T)! ordered assignments. For Q=5, T=4 this is
    only 120 permutations — trivially parallel on GPU, eliminating the
    GPU→CPU→GPU transfer incurred by the scipy-based Matcher every step.

    Drop-in replacement for Matcher: identical forward() signature and output
    contract. Falls back to scipy for events where T > max_targets.

    Parameters
    ----------
    num_queries : int
        Number of decoder query slots (Q). Permutations are pre-generated for
        this Q at construction time.
    max_targets : int
        Maximum number of targets (T) supported on GPU. Must satisfy
        P(Q, max_targets) is feasible in GPU memory. Default 5.
    """

    def __init__(self, num_queries: int, max_targets: int = 5):
        super().__init__()
        self.num_queries = num_queries
        self.max_targets = max_targets
        self._logged_first_call = False

        # Pre-generate and register permutation tensors for each t in 1..max_targets.
        # Shape of perms_t: [P(Q, t), t]  (stored as non-persistent buffers)
        from itertools import permutations as _permutations
        for t in range(1, max_targets + 1):
            perms = list(_permutations(range(num_queries), t))
            buf = torch.tensor(perms, dtype=torch.long)  # [P(Q,t), t]
            self.register_buffer(f"perms_{t}", buf, persistent=False)

        print(f"[BruteForceGPUMatcher] Created with Q={num_queries}, max_targets={max_targets}")

    def _get_perms(self, t: int) -> torch.Tensor:
        return getattr(self, f"perms_{t}")

    def _match_group(
        self,
        group_costs: torch.Tensor,  # [n, Q, t]
        t: int,
    ) -> torch.Tensor:
        """Return pred_idxs [n, Q] for a group of n events all having t targets."""
        n, Q, _ = group_costs.shape
        device = group_costs.device
        perms = self._get_perms(t).to(device)  # [num_perms, t]
        num_perms = perms.shape[0]

        # Vectorised cost sum over all permutations — no Python loop.
        # group_costs: [n, Q, t] → permute to [n, t, Q]
        # perms.T:     [t, num_perms] → expand to [n, t, num_perms]
        # gather along Q dim → [n, t, num_perms], then sum over t → [n, num_perms]
        costs_tq = group_costs.permute(0, 2, 1)                        # [n, t, Q]
        perms_exp = perms.T.unsqueeze(0).expand(n, -1, -1)             # [n, t, num_perms]
        total_cost = costs_tq.gather(2, perms_exp).sum(dim=1)          # [n, num_perms]

        # Best permutation per event
        best_idx = total_cost.argmin(dim=1)          # [n]
        matched_queries = perms[best_idx]            # [n, t]  — query idx per target

        # Build unmatched query indices (ascending) using a sort trick:
        # Give matched positions a large offset so they sort to the end.
        all_q = torch.arange(Q, device=device).unsqueeze(0).expand(n, -1)  # [n, Q]
        is_matched = torch.zeros(n, Q, dtype=torch.bool, device=device)
        is_matched.scatter_(1, matched_queries, True)
        sort_key = all_q + Q * is_matched.long()     # unmatched: 0..Q-1, matched: Q..2Q-1
        sorted_keys, _ = sort_key.sort(dim=1)
        unmatched = sorted_keys[:, : Q - t]          # [n, Q-t]  sorted unmatched indices

        pred_idxs = torch.cat([matched_queries, unmatched], dim=1)  # [n, Q]
        return pred_idxs

    @torch.no_grad()
    def forward(
        self,
        costs: torch.Tensor,
        object_valid_mask: torch.Tensor = None,
        query_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Match predictions to targets on GPU via brute-force permutation search.

        Fully vectorised — no Python loops over batch or T values.

        Parameters
        ----------
        costs : torch.Tensor [B, Q, T]
        object_valid_mask : torch.Tensor [B, T] bool, optional
        query_valid_mask : torch.Tensor [B, Q] bool, optional

        Returns
        -------
        pred_idxs : torch.Tensor [B, Q] int64
            First T_i entries per row are matched query indices (in target order).
            Remaining Q-T_i entries are unmatched query indices in ascending order.
        """
        # Cast to float32 for numerical stability (cost matrix may arrive as bf16
        # under mixed-precision training; bf16's ~3 decimal digits can cause wrong
        # argmin results when permutation costs are close).
        costs = costs.detach().float()

        B, Q, T_max = costs.shape
        device = costs.device

        if not self._logged_first_call:
            self._logged_first_call = True
            print(f"[BruteForceGPUMatcher] GPU path active — B={B}, Q={Q}, T_max={T_max}, device={device}")

        if object_valid_mask is None:
            object_valid_mask = torch.ones(B, T_max, dtype=torch.bool, device=device)

        if query_valid_mask is not None:
            costs = costs.masked_fill(~query_valid_mask.unsqueeze(-1), 1e9)

        if T_max == 0:
            return torch.arange(Q, device=device).unsqueeze(0).expand(B, -1).contiguous()

        per_event_T = object_valid_mask.sum(dim=1)

        if T_max > self.max_targets:
            warnings.warn(
                f"BruteForceGPUMatcher: T_max={T_max} > max_targets={self.max_targets}. "
                "Falling back to scipy for the whole batch.",
                RuntimeWarning, stacklevel=2,
            )
            target_order = torch.argsort(
                object_valid_mask.to(torch.int8), dim=1, descending=True
            )
            costs = torch.gather(
                costs, 2, target_order.unsqueeze(1).expand(B, Q, T_max)
            )
            max_valid = int(per_event_T.max().item())
            costs = costs[:, :, :max_valid]
            return self._scipy_fallback(
                costs.cpu().numpy(), Q, per_event_T.cpu().numpy()
            ).to(device)

        # Compact arbitrary valid target columns to the prefix consumed by the
        # permutation code.  Keeping this in the GPU path makes it agree with
        # the scipy matcher for masks such as [False, True].
        target_order = torch.argsort(
            object_valid_mask.to(torch.int8), dim=1, descending=True
        )
        gather_order = target_order.unsqueeze(1).expand(B, Q, T_max)
        costs = torch.gather(costs, 2, gather_order)
        object_valid_mask = torch.arange(T_max, device=device).unsqueeze(0) < per_event_T.unsqueeze(1)

        # Zero phantom target columns so they contribute 0 to the cost sum.
        # The optimal assignment for real targets is unaffected.
        valid_costs = costs * object_valid_mask.unsqueeze(1).float()  # [B, Q, T_max]

        # Score every P(Q, T_max) permutation for every event — single gather+sum, no loop.
        # valid_costs [B, Q, T_max] → permute → [B, T_max, Q]
        # perms.T expanded:          [B, T_max, num_perms]
        # gather along Q dim      → [B, T_max, num_perms], sum over T_max → [B, num_perms]
        perms     = self._get_perms(T_max).to(device)              # [num_perms, T_max]
        costs_tq  = valid_costs.permute(0, 2, 1)                   # [B, T_max, Q]
        perms_exp = perms.T.unsqueeze(0).expand(B, -1, -1)         # [B, T_max, num_perms]
        total_cost = costs_tq.gather(2, perms_exp).sum(dim=1)      # [B, num_perms]

        best_idx  = total_cost.argmin(dim=1)  # [B]
        best_perm = perms[best_idx]           # [B, T_max] — query assigned to each target slot

        # Build a sort key for every query:
        #   real-matched query (assigned to target j):  key = j      (sorts to position j)
        #   unmatched / phantom query q:                key = Q + q  (sorts after real, by index)
        pos_is_real = (
            torch.arange(T_max, device=device).unsqueeze(0) < per_event_T.unsqueeze(1)
        )  # [B, T_max]

        # Redirect phantom target slots to a sentinel column (Q) so their scatter writes
        # are discarded without affecting the real query keys.
        best_perm_safe  = best_perm.masked_fill(~pos_is_real, Q)           # [B, T_max]
        target_pos      = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
        target_pos_safe = target_pos.masked_fill(~pos_is_real, 0)          # phantom → sentinel col

        # Default key: unmatched query q gets Q + q
        query_key = torch.arange(Q, device=device).unsqueeze(0).expand(B, -1).clone() + Q
        # +1 sentinel column absorbs phantom scatter writes
        query_key_ext = torch.cat([query_key, query_key.new_zeros(B, 1)], dim=1)
        query_key_ext.scatter_(1, best_perm_safe, target_pos_safe)
        query_key = query_key_ext[:, :Q]  # [B, Q]

        _, pred_idxs = query_key.sort(dim=1)

        assert (pred_idxs >= 0).all(), "BruteForceGPUMatcher: negative index produced"
        return pred_idxs

    @staticmethod
    def _scipy_fallback(
        costs_np: np.ndarray, Q: int, lengths=None
    ) -> torch.Tensor:
        """Sequential scipy fallback for T > max_targets edge cases.

        costs_np has shape [B, Q, T]. The Matcher/solve_scipy convention
        expects [T, Q] (targets as rows, queries as columns) so that
        col_idx from linear_sum_assignment returns query indices.
        """
        # Transpose: [B, Q, T] -> [B, T, Q]
        costs_t = costs_np.swapaxes(1, 2)
        results = []
        default_idx = np.arange(Q, dtype=np.int32)
        for b in range(len(costs_t)):
            length = costs_t.shape[1] if lengths is None else int(lengths[b])
            idx = match_individual(SOLVERS["scipy"], costs_t[b][:length], default_idx)
            results.append(idx)
        return torch.from_numpy(np.stack(results))


def create_matcher(
    matching_solver: str = "gpu_bruteforce",
    num_queries: int = 5,
    max_targets: int = 5,
    # scipy / Matcher kwargs
    adaptive_solver: bool = True,
    adaptive_check_interval: int = 1000,
    parallel_solver: bool = True,
    parallel_backend: str = "thread",
    n_jobs: int = 8,
    verbose: bool = False,
) -> nn.Module:
    """Factory that returns a Matcher or BruteForceGPUMatcher.

    Parameters
    ----------
    matching_solver : str
        ``"gpu_bruteforce"`` (default) — GPU-resident brute-force matcher.
        Any scipy solver name (``"scipy"``) — original CPU-based Matcher.
    num_queries : int
        Number of decoder query slots Q. Only used by gpu_bruteforce.
    max_targets : int
        GPU fallback threshold. Only used by gpu_bruteforce.
    """
    if matching_solver == "gpu_bruteforce":
        return BruteForceGPUMatcher(num_queries=num_queries, max_targets=max_targets)
    return Matcher(
        default_solver=matching_solver,
        adaptive_solver=adaptive_solver,
        adaptive_check_interval=adaptive_check_interval,
        parallel_solver=parallel_solver,
        parallel_backend=parallel_backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
