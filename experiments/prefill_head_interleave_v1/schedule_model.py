"""Idealized greedy CTA scheduling: tile count only, no cache or launch overhead."""
import heapq

QUERY_LENGTH, QUERY_TILE, KEY_TILE, SMS = 16384, 24, 128, 132
nqb = (QUERY_LENGTH + QUERY_TILE - 1) // QUERY_TILE
for context in (0, 32768, 65536):
    costs = [(context + min((q + 1) * QUERY_TILE, QUERY_LENGTH) + KEY_TILE - 1) // KEY_TILE
             for q in range(nqb)]
    for label, jobs in (('separate', list(reversed(costs)) * 2),
                        ('interleaved', [c for c in reversed(costs) for _ in range(2)])):
        workers = [0] * SMS
        for cost in jobs:
            ready = heapq.heappop(workers)
            heapq.heappush(workers, ready + cost)
        print(f'context={context} order={label} makespan_tiles={max(workers)} '
              f'mean_tiles={sum(workers) / SMS:.2f}')
