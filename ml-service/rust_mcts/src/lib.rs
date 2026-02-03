use pyo3::prelude::*;

mod node;
mod tree;
mod game_state;

pub use node::MCTSNode;
pub use tree::MCTSTree;
pub use game_state::GameState;

/// Rust-accelerated MCTS for AlphaZero
#[pymodule]
fn rust_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MCTSTree>()?;
    m.add_class::<GameState>()?;
    m.add_function(wrap_pyfunction!(batch_select_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(batch_expand_and_backprop, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_tree_ops, m)?)?;
    Ok(())
}

/// Select multiple leaves from multiple trees in parallel
#[pyfunction]
fn batch_select_leaves(
    trees: Vec<PyRef<MCTSTree>>,
    c_puct: f32,
    batch_size: usize,
) -> PyResult<Vec<(usize, Vec<u32>, Vec<f32>)>> {
    // Returns: [(tree_idx, path_indices, move_priors), ...]
    let mut results = Vec::with_capacity(trees.len() * batch_size);

    for (tree_idx, tree) in trees.iter().enumerate() {
        for _ in 0..batch_size {
            if let Some((path, priors)) = tree.select_leaf(c_puct) {
                results.push((tree_idx, path, priors));
            }
        }
    }

    Ok(results)
}

/// Expand nodes and backpropagate values in batch
#[pyfunction]
fn batch_expand_and_backprop(
    trees: &Bound<'_, pyo3::types::PyList>,
    expansions: Vec<(usize, Vec<u32>, Vec<(u32, f32)>, f32)>,
) -> PyResult<()> {
    // expansions: [(tree_idx, path, [(move_id, prior), ...], value), ...]
    for (tree_idx, path, children_priors, value) in expansions {
        let tree: PyRef<MCTSTree> = trees.get_item(tree_idx)?.extract()?;
        tree.expand_and_backprop_internal(&path, &children_priors, value);
    }
    Ok(())
}

/// Benchmark pure tree operations (no Python overhead in the loop)
#[pyfunction]
fn benchmark_tree_ops(num_simulations: usize, num_children: usize, c_puct: f32) -> PyResult<(f64, f64, f64)> {
    use std::time::Instant;

    // Create a tree
    let tree = MCTSTree::new(0);

    // Benchmark select
    let start = Instant::now();
    for _ in 0..num_simulations {
        let _ = tree.select_leaf(c_puct);
    }
    let select_time = start.elapsed().as_secs_f64();

    // Benchmark expand
    let children: Vec<(u32, f32)> = (0..num_children as u32)
        .map(|i| (i, 1.0 / num_children as f32))
        .collect();

    let start = Instant::now();
    for _i in 0..num_simulations {
        let path: Vec<u32> = vec![0];
        tree.expand_and_backprop_internal(&path, &children, 0.5);
    }
    let expand_time = start.elapsed().as_secs_f64();

    // Benchmark full cycle
    let tree2 = MCTSTree::new(0);
    let start = Instant::now();
    for i in 0..num_simulations {
        let _leaf = tree2.select_leaf(c_puct);
        let path: Vec<u32> = vec![0];
        tree2.expand_and_backprop_internal(&path, &children, if i % 2 == 0 { 1.0 } else { -1.0 });
    }
    let full_time = start.elapsed().as_secs_f64();

    Ok((select_time, expand_time, full_time))
}
