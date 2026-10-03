import json
import math
import numpy as np
import pytest

from systolic_sim import SystolicSimulator
from run_all import FIXED_LAYERS, main as run_all_main


def test_worked_example_reads_writes():
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, (12, 4))
    B = rng.uniform(-1, 1, (4, 8))

    expected_reads = {'WS': 128, 'OS': 192, 'RS': 144}
    for dataflow, exp_reads in expected_reads.items():
        sim = SystolicSimulator(4, 4, sram_size=10**9, sram_bw=32)
        res, cycles, util = sim.matmul(A, B, dataflow=dataflow, double_buffer=False)
        stats = sim.get_memory_stats()
        assert np.allclose(res, A @ B, rtol=1e-9, atol=1e-9)
        assert stats['sram_reads'] == exp_reads
        assert stats['sram_writes'] == 96


def test_worked_example_cycles():
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, (12, 4))
    B = rng.uniform(-1, 1, (4, 8))

    expected_db_false = {'WS': 98, 'OS': 114, 'RS': 102}
    expected_db_true = {'WS': 74, 'OS': 74, 'RS': 74}

    for dataflow in ('WS', 'OS', 'RS'):
        sim = SystolicSimulator(4, 4, sram_size=10**9, sram_bw=32)
        _, cycles_false, _ = sim.matmul(A, B, dataflow=dataflow, double_buffer=False)
        assert cycles_false == expected_db_false[dataflow]

        sim_true = SystolicSimulator(4, 4, sram_size=10**9, sram_bw=32)
        _, cycles_true, _ = sim_true.matmul(A, B, dataflow=dataflow, double_buffer=True)
        assert cycles_true == expected_db_true[dataflow]


@pytest.mark.parametrize("M,K,N,R,C", [
    (16, 16, 16, 4, 4),
    (10, 7, 13, 4, 4),
    (3, 5, 3, 8, 8),
    (20, 20, 20, 4, 8),
])
def test_section_viii_shapes(M, K, N, R, C):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K))
    B = rng.uniform(-1, 1, (K, N))

    for dataflow in ('WS', 'OS', 'RS'):
        for db in (False, True):
            sim = SystolicSimulator(R, C, sram_size=10**9, sram_bw=64)
            res, cycles, util = sim.matmul(A, B, dataflow=dataflow, double_buffer=db)

            assert np.allclose(res, A @ B, rtol=1e-9, atol=1e-9)

            stats = sim.get_memory_stats()
            pemap = sim.get_pe_activity_map()

            assert pemap.shape == (R, C)
            assert pemap.sum() == M * N * K
            assert stats['sram_writes'] == M * N
            assert 0.0 <= util <= 1.0 + 1e-9


def test_invalid_parameters_raise():
    sim = SystolicSimulator(4, 4)

    with pytest.raises(ValueError):
        sim.matmul(np.zeros((4, 4)), np.zeros((4, 4)), dataflow='INVALID')

    with pytest.raises(ValueError):
        sim.matmul(np.zeros((4, 8)), np.zeros((4, 4)))

    with pytest.raises(ValueError):
        sim.matmul(np.zeros((4, 4, 4)), np.zeros((4, 4)))

    with pytest.raises(ValueError):
        SystolicSimulator(0, 4)


def test_capacity_check_raises_memoryerror():
    sim = SystolicSimulator(64, 64, sram_size=1000)
    with pytest.raises(MemoryError):
        sim.matmul(np.zeros((64, 64)), np.zeros((64, 64)))


def test_state_reset_between_calls():
    sim = SystolicSimulator(4, 4, sram_size=10**9, sram_bw=32)
    A1, B1 = np.ones((4, 4)), np.ones((4, 4))
    A2, B2 = np.ones((8, 8)), np.ones((8, 8))

    sim.matmul(A1, B1)
    stats1 = sim.get_memory_stats()

    sim.matmul(A2, B2)
    stats2 = sim.get_memory_stats()

    assert stats2['sram_writes'] == 64
    assert stats2['sram_writes'] != stats1['sram_writes']


def test_results_json_structure_and_self_consistency():
    run_all_main()

    with open("results.json", "r") as f:
        data = json.load(f)

    assert "level4" in data
    lvl4 = data["level4"]
    assert "traffic_by_dataflow" in lvl4
    assert "cycles_by_dataflow" in lvl4
    assert "best_dataflow_by_traffic" in lvl4

    for layer in FIXED_LAYERS:
        assert layer in lvl4["traffic_by_dataflow"]
        assert layer in lvl4["cycles_by_dataflow"]
        assert layer in lvl4["best_dataflow_by_traffic"]
        best_df = lvl4["best_dataflow_by_traffic"][layer]
        assert best_df in ("WS", "OS", "RS")

    assert "level5" in data
    lvl5 = data["level5"]
    assert "roofline_points" in lvl5
    assert "ridge_point_AI" in lvl5
    assert "best_config" in lvl5

    assert math.isclose(lvl5["ridge_point_AI"], 128.0, rel_tol=1e-6)
    assert lvl5["best_config"]["array"] in (8, 16, 32, 64, 128)
    assert lvl5["best_config"]["sram_bw"] in (32, 64, 128, 256, 512)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
