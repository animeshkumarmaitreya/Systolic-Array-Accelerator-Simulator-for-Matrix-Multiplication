import json
import numpy as np
from systolic_sim import SystolicSimulator

FIXED_LAYERS = {
    "resnet_stem": (1, 3 * 7 * 7, 64),
    "resnet_block1": (1, 64, 64),
    "resnet_block3": (1, 256, 512),
    "resnet_fc": (1, 2048, 1000),
}

DATAFLOWS = ("WS", "OS", "RS")


def make_matrices(M, K, N, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, size=(M, K))
    B = rng.uniform(-1, 1, size=(K, N))
    return A, B


def compute_level4():
    R = C = 16
    sram_bw = 128
    sram_size = 1 << 30

    traffic_by_dataflow = {}
    cycles_by_dataflow = {}
    best_dataflow_by_traffic = {}

    for layer_name, (M, K, N) in FIXED_LAYERS.items():
        A, B = make_matrices(M, K, N)
        traffic_by_dataflow[layer_name] = {}
        cycles_by_dataflow[layer_name] = {}

        for dataflow in DATAFLOWS:
            sim = SystolicSimulator(R, C, sram_size=sram_size, sram_bw=sram_bw)
            _, cycles, _ = sim.matmul(A, B, dataflow=dataflow, double_buffer=True)
            stats = sim.get_memory_stats()
            traffic = stats["sram_reads"] + stats["sram_writes"]

            traffic_by_dataflow[layer_name][dataflow] = int(traffic)
            cycles_by_dataflow[layer_name][dataflow] = int(cycles)

        best_dataflow_by_traffic[layer_name] = min(
            DATAFLOWS, key=lambda df: traffic_by_dataflow[layer_name][df]
        )

    return {
        "traffic_by_dataflow": traffic_by_dataflow,
        "cycles_by_dataflow": cycles_by_dataflow,
        "best_dataflow_by_traffic": best_dataflow_by_traffic,
    }


def compute_level5():
    array_sizes = [8, 16, 32, 64, 128]
    bandwidths = [32, 64, 128, 256, 512]
    sram_size = 1 << 30

    roofline_points = []
    cycles_by_config = {}

    for array in array_sizes:
        for bw in bandwidths:
            total_cycles_sum = 0
            for layer_name, (M, K, N) in FIXED_LAYERS.items():
                A, B = make_matrices(M, K, N)
                sim = SystolicSimulator(array, array, sram_size=sram_size, sram_bw=bw)
                _, cycles, _ = sim.matmul(A, B, dataflow="WS", double_buffer=True)
                stats = sim.get_memory_stats()

                macs = M * K * N
                bytes_moved = 8 * (stats["sram_reads"] + stats["sram_writes"])
                AI = macs / bytes_moved
                P = macs / cycles

                roofline_points.append({
                    "layer": layer_name,
                    "array": array,
                    "sram_bw": bw,
                    "AI": float(AI),
                    "P": float(P),
                })

                total_cycles_sum += cycles

            cycles_by_config[(array, bw)] = total_cycles_sum

    peak_compute_macs_per_cycle = 128 * 128
    ridge_point_AI = peak_compute_macs_per_cycle / 128.0

    best_array, best_bw = min(cycles_by_config, key=lambda k: cycles_by_config[k])

    return {
        "roofline_points": roofline_points,
        "ridge_point_AI": float(ridge_point_AI),
        "best_config": {"array": int(best_array), "sram_bw": int(best_bw)},
    }


def main():
    results = {
        "level4": compute_level4(),
        "level5": compute_level5(),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
