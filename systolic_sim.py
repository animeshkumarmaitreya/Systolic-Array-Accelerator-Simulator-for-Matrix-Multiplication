import math
import numpy as np


class SystolicSimulator:
    def __init__(self, array_rows: int, array_cols: int,
                 sram_size: int = 65536, sram_bw: int = 128,
                 pe_regfile_size: int = 4):
        if array_rows < 1 or array_cols < 1:
            raise ValueError("array_rows and array_cols must be >= 1")
        if sram_size < 1 or sram_bw < 1:
            raise ValueError("sram_size and sram_bw must be >= 1")

        self.array_rows = array_rows
        self.array_cols = array_cols
        self.sram_size = sram_size
        self.sram_bw = sram_bw
        self.pe_regfile_size = pe_regfile_size

        self._pe_activity = None
        self._memory_stats = None

    def matmul(self, A: np.ndarray, B: np.ndarray,
               dataflow: str = 'WS', double_buffer: bool = False
              ) -> tuple[np.ndarray, int, float]:
        if dataflow not in ('WS', 'OS', 'RS'):
            raise ValueError(f"Invalid dataflow: {dataflow}")

        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Matrices A and B must be 2D")

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Inner dimension mismatch: {A.shape} vs {B.shape}")

        R, C = self.array_rows, self.array_cols

        bytes_tile = 8 * (R * K + K * C + R * C)
        if bytes_tile > self.sram_size:
            raise MemoryError(
                f"Working set {bytes_tile} bytes exceeds SRAM size {self.sram_size}"
            )

        TM = math.ceil(M / R)
        TN = math.ceil(N / C)

        def row_size(i):
            return min(R, M - i * R)

        def col_size(j):
            return min(C, N - j * C)

        if dataflow == 'WS':
            order = [(i, j) for j in range(TN) for i in range(TM)]
        elif dataflow == 'RS':
            order = [(i, j) for i in range(TM) for j in range(TN)]
        else:
            order = [(i, j) for i in range(TM) for j in range(TN)]

        pe_activity = np.zeros((R, C), dtype=np.int64)
        tile_cycles_list = []
        delta_list = []
        reads_total = 0
        writes_total = 0
        active_pe_cycle_numerator = 0

        seen_j = set()
        seen_i = set()

        for (i, j) in order:
            r = row_size(i)
            c = col_size(j)

            cycles_tile = (r - 1) + (c - 1) + K + 1
            tile_cycles_list.append(cycles_tile)

            writes_total += r * c
            active_pe_cycle_numerator += r * c * K
            pe_activity[:r, :c] += K

            if dataflow == 'WS':
                a_new = r * K
                if j not in seen_j:
                    b_new = K * c
                    seen_j.add(j)
                else:
                    b_new = 0
                delta = a_new + b_new
            elif dataflow == 'RS':
                b_new = K * c
                if i not in seen_i:
                    a_new = r * K
                    seen_i.add(i)
                else:
                    a_new = 0
                delta = a_new + b_new
            else:
                delta = r * K + K * c

            reads_total += delta
            delta_list.append(delta)

        T = len(order)
        stalls = [0] * T
        peak_bytes = 0

        for t in range(T):
            byte_req = 8 * delta_list[t]
            peak_bytes = max(peak_bytes, byte_req)
            Lt = math.ceil(byte_req / self.sram_bw)

            if t == 0:
                stalls[t] = Lt
            elif not double_buffer:
                stalls[t] = Lt
            else:
                stalls[t] = max(0, Lt - tile_cycles_list[t - 1])

        total_cycles = sum(stalls) + sum(tile_cycles_list)
        utilization = active_pe_cycle_numerator / (total_cycles * R * C)

        result = A @ B

        self._pe_activity = pe_activity
        self._memory_stats = {
            'sram_reads': int(reads_total),
            'sram_writes': int(writes_total),
            'stall_cycles': int(sum(stalls)),
            'peak_sram_bw_used': int(peak_bytes),
        }

        return result, int(total_cycles), float(utilization)

    def get_pe_activity_map(self) -> np.ndarray:
        if self._pe_activity is None:
            raise RuntimeError("matmul() must be called first")
        return self._pe_activity.copy()

    def get_memory_stats(self) -> dict:
        if self._memory_stats is None:
            raise RuntimeError("matmul() must be called first")
        return dict(self._memory_stats)
