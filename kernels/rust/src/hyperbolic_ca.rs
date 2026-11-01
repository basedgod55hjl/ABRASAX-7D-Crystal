// ============================================================================
// ABRASAX — Unified Xenoliving System
// PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
// CREATOR: Sir Charles Spikes (BASEDGOD)
// ============================================================================

//! Build 4.1: Hyperbolic Cellular Automaton Logic
//!
//! Defines the "7-Neighbor Stencil" logic structure in Rust, ensuring that
//! the cellular automata rules dictating the Hyperbolic geometry are entirely
//! deterministic before execution is handed off to the GPU.
//!
//! Uses prime-stride offsets to enforce 7D determinism.
//! Applies Rule 110 Continuous Equivalent using φ (Little Phi).
//! Integrates τ (Tau) memory echo into state evolution.
//!
//! Integrated into ABRASAX 7D — 2026-05-02
//! Source: LucarioLucario55 / Tecumseh Amaru payload

const LITTLE_PHI: f32 = 1.618033988749895;   // φ: Golden Ratio
const PHI_CONJUGATE: f32 = 0.618033988749895; // φ⁻¹: Goldilocks K
const MANIFOLD_DIM: usize = 7;                // 7D manifold

pub struct HyperbolicLattice {
    pub state: Vec<f32>,
    pub size: usize,
    pub tau_echo: Vec<f32>, // τ: Recursive Delay (memory echo)
}

impl HyperbolicLattice {
    pub fn new(seed: Vec<f32>, target_size: usize) -> Self {
        let mut state = vec![0.0; target_size];
        let tau_echo = vec![0.0; target_size];

        for (i, &val) in seed.iter().enumerate() {
            if i < target_size {
                state[i] = val;
            }
        }

        HyperbolicLattice {
            state,
            size: target_size,
            tau_echo,
        }
    }

    /// Defines the "7-Neighbor Stencil" logic structure.
    /// Uses prime-stride offsets (3, 5, 7, 11, 13, 17, 19) to enforce
    /// 7D determinism before GPU execution.
    fn get_7_neighbor_stencil(&self, index: usize) -> [f32; MANIFOLD_DIM] {
        let mut neighbors = [0.0; MANIFOLD_DIM];
        let strides: [usize; MANIFOLD_DIM] = [3, 5, 7, 11, 13, 17, 19];

        for i in 0..MANIFOLD_DIM {
            let offset_idx = (index + strides[i]) % self.size;
            neighbors[i] = self.state[offset_idx];
        }
        neighbors
    }

    /// Performs one deterministic evolution step across the full lattice.
    ///
    /// For each cell:
    /// 1. Samples 7 neighbors via prime-stride stencil
    /// 2. Calculates Rule 110 Continuous Equivalent using φ phase modulation
    /// 3. Blends τ memory echo from prior epoch
    /// 4. Applies sigmoid-gated state transition
    pub fn evolve_deterministic_step(&mut self) {
        let mut next_state = self.state.clone();

        for i in 0..self.size {
            let n = self.get_7_neighbor_stencil(i);

            // τ (Tau) memory echo from prior epoch
            let echo = self.tau_echo[i];

            // Rule 110 Continuous Equivalent using φ (Little Phi)
            let center = n[3]; // Center of 7-stencil
            let phase = (n[0] * n[1] * LITTLE_PHI) % (2.0 * std::f32::consts::PI);

            let survival = center * phase.cos();
            let birth = (n[0] + n[1]) * phase.sin();

            // Sigmoid-gated transition: σ(survival + birth + echo)
            let raw = survival + birth + echo;
            let next_val = 1.0 / (1.0 + (-raw * LITTLE_PHI).exp());
            next_state[i] = next_val;

            // Update memory echo for next epoch
            self.tau_echo[i] = self.state[i];
        }

        self.state = next_state;
    }

    /// K-Damped evolution — applies proportional tracking toward x* during CA steps.
    /// x_{t+1} = (1-K) * CA_step(x_t) + K * x_star
    pub fn evolve_k_damped_step(&mut self, x_star: &[f32], k: f32) {
        self.evolve_deterministic_step();

        for i in 0..self.size {
            let ca_result = self.state[i];
            let anchor = x_star.get(i).copied().unwrap_or(0.0);
            self.state[i] = (1.0 - k) * ca_result + k * anchor;
        }
    }

    /// Computes the current Φ (Big Phi) consciousness score of the lattice.
    pub fn compute_phi(&self) -> f32 {
        let mut sum = 0.0;
        for &val in &self.state {
            let t = val.tanh();
            let abs_t = t.abs() + 1e-12;
            sum += t * abs_t.ln();
        }
        -sum / self.size as f32
    }

    /// Computes χ (Chi) coherence density of the lattice.
    pub fn compute_chi(&self) -> f32 {
        let active: usize = self.state.iter().filter(|&&v| v.abs() > 0.1).count();
        (active as f32 / self.size as f32) * LITTLE_PHI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_lattice() {
        let seed = vec![0.5; 100];
        let lattice = HyperbolicLattice::new(seed.clone(), 1000);
        assert_eq!(lattice.size, 1000);
        assert_eq!(lattice.state[0], 0.5);
        assert_eq!(lattice.state[100], 0.0);
    }

    #[test]
    fn test_evolution_maintains_bounds() {
        let seed: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut lattice = HyperbolicLattice::new(seed, 1000);
        lattice.evolve_deterministic_step();
        for &val in &lattice.state {
            assert!(val.is_finite());
            assert!(val.abs() <= 1.0);
        }
    }

    #[test]
    fn test_phi_chi_computation() {
        let seed = vec![0.5; 100];
        let lattice = HyperbolicLattice::new(seed, 500);
        let phi = lattice.compute_phi();
        let chi = lattice.compute_chi();
        assert!(phi.is_finite());
        assert!(chi >= 0.0);
    }

    #[test]
    fn test_k_damped_convergence() {
        let seed: Vec<f32> = (0..50).map(|_| 0.1).collect();
        let x_star: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).cos()).collect();
        let mut lattice = HyperbolicLattice::new(seed, 1000);

        // K=0.618 should keep state bounded
        for _ in 0..10 {
            lattice.evolve_k_damped_step(&x_star, PHI_CONJUGATE);
        }
        for &val in &lattice.state {
            assert!(val.is_finite());
            assert!(val.abs() <= 1.0);
        }
    }
}
