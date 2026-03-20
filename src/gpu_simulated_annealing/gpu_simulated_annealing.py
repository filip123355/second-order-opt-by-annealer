import subprocess
import tempfile
from pathlib import Path

import dimod
import numpy as np
from dimod import BinaryQuadraticModel


class GPUSimulatedAnnealingSampler:
    def __init__(self, 
                 exe: str="engine", 
                 steps: int=200, 
                 beta_start: float=0.1, 
                 beta_end: float=5.0,
    ) -> None:
        module_dir = Path(__file__).resolve().parent
        exe_path = Path(exe)
        self.exe = str(exe_path if exe_path.is_absolute() else (module_dir / exe_path))
        self.steps = int(steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

    def sample(self, 
               bqm: BinaryQuadraticModel, 
               num_reads: int = 100,
    ) -> dimod.SampleSet:
        
        ising_bqm = bqm.change_vartype(dimod.SPIN, inplace=False) # convert sample to SPIN for GPU annealer

        variables = list(ising_bqm.variables)
        N = len(variables)
        idx = {v: i for i, v in enumerate(variables)}

        J = np.zeros((N, N), dtype=np.float32)
        h = np.zeros(N, dtype=np.float32)

        for v, bias in ising_bqm.linear.items():
            h[idx[v]] = np.float32(bias)

        for (u, v), bias in ising_bqm.quadratic.items():
            i, j = idx[u], idx[v]
            J[i, j] = np.float32(bias)
            J[j, i] = np.float32(bias)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            j_path = tmpdir / "j.bin"
            h_path = tmpdir / "h.bin"
            e_path = tmpdir / "bestE.bin"
            s_path = tmpdir / "bestSample.bin"

            J.tofile(j_path)
            h.tofile(h_path)
            
            # Computation on CUDA engine
            subprocess.run(
                [
                    self.exe,
                    str(N),
                    str(j_path),
                    str(h_path),
                    str(int(num_reads)),
                    str(self.steps),
                    str(self.beta_start),
                    str(self.beta_end),
                ],
                check=True,
            )

            best_e = np.fromfile(e_path, dtype=np.float32)
            best_spins = np.fromfile(s_path, dtype=np.int8).reshape(int(num_reads), N)

        r = int(np.argmin(best_e))
        spin_sample = {variables[i]: int(best_spins[r, i]) for i in range(N)}
        # Convert spins back to original vartype (BINARY expected by optimizer).
        binary_sample = {
            v: 1 if spin_sample[v] == 1 else 0
            for v in variables
        }
        energy = float(bqm.energy(binary_sample))

        return dimod.SampleSet.from_samples(
            [binary_sample],
            vartype=dimod.BINARY,
            energy=[energy],
        )