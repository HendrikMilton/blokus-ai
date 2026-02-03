use crate::node::MCTSNode;
use parking_lot::RwLock;
use pyo3::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};

/// MCTS Tree with fast Rust operations
#[pyclass]
pub struct MCTSTree {
    /// All nodes in the tree (arena allocation)
    nodes: RwLock<Vec<MCTSNode>>,

    /// Next node ID
    next_id: AtomicU32,

    /// Root node index (always 0)
    root: u32,

    /// Current player at root
    root_player: u32,
}

// Internal methods (not exposed to Python)
impl MCTSTree {
    /// Backpropagate value up the tree
    fn backpropagate_internal(&self, path: &[u32], value: f32) {
        let nodes = self.nodes.read();
        let mut current_value = value;

        // Backprop from leaf to root, alternating sign for opponent
        for (i, &node_idx) in path.iter().rev().enumerate() {
            let node = &nodes[node_idx as usize];

            // Remove virtual loss if applied during selection
            if i > 0 {
                node.remove_virtual_loss();
            }

            // Add visit and value
            node.backprop(current_value);

            // Flip value for opponent's perspective
            current_value = -current_value;
        }
    }

    /// Internal expand and backprop (takes references)
    pub fn expand_and_backprop_internal(&self, path: &[u32], children_priors: &[(u32, f32)], value: f32) {
        if path.is_empty() {
            return;
        }

        let leaf_idx = *path.last().unwrap();

        // Create child nodes if priors provided
        if !children_priors.is_empty() {
            let nodes = self.nodes.write();
            let leaf = &nodes[leaf_idx as usize];

            // Store priors for later expansion
            leaf.set_child_priors(children_priors);

            // Mark as expanded with priors stored
            if !leaf.is_expanded() {
                leaf.expand(&[]);
            }
        }

        // Backpropagate
        self.backpropagate_internal(path, value);
    }
}

#[pymethods]
impl MCTSTree {
    #[new]
    pub fn new(root_player: u32) -> Self {
        let root_node = MCTSNode::new(0, 0, 1.0, u32::MAX);
        Self {
            nodes: RwLock::new(vec![root_node]),
            next_id: AtomicU32::new(1),
            root: 0,
            root_player,
        }
    }

    /// Get number of nodes in tree
    pub fn size(&self) -> usize {
        self.nodes.read().len()
    }

    /// Get root visit count
    pub fn root_visits(&self) -> u32 {
        self.nodes.read()[0].visit_count()
    }

    /// Set root policy (child priors)
    pub fn set_root_policy(&self, moves_and_priors: Vec<(u32, f32)>) {
        let nodes = self.nodes.read();
        nodes[0].set_child_priors(&moves_and_priors);
    }

    /// Select a leaf node, returning the path and unexpanded move priors if any
    /// Returns: (path_indices, child_priors_at_leaf)
    pub fn select_leaf(&self, c_puct: f32) -> Option<(Vec<u32>, Vec<f32>)> {
        let nodes = self.nodes.read();
        let mut path = Vec::with_capacity(32);
        let mut current_idx = self.root;

        loop {
            path.push(current_idx);
            let node = &nodes[current_idx as usize];

            // If not expanded, return with priors for expansion
            if !node.is_expanded() {
                let priors: Vec<f32> = node.get_child_priors()
                    .iter()
                    .map(|(_, p)| *p)
                    .collect();
                return Some((path, priors));
            }

            let children = node.get_children();
            if children.is_empty() {
                // Terminal node
                return Some((path, vec![]));
            }

            // Get parent visits for UCB calculation
            let parent_visits = node.visit_count();

            // Find best child considering both expanded and unexpanded
            let mut best_score = f32::NEG_INFINITY;
            let mut best_child_idx: Option<u32> = None;
            let mut best_unexpanded: Option<(u32, f32)> = None;

            // Check expanded children
            for &(_move_id, child_idx) in children.iter() {
                let child = &nodes[child_idx as usize];
                let score = child.ucb_score(parent_visits, c_puct);
                if score > best_score {
                    best_score = score;
                    best_child_idx = Some(child_idx);
                    best_unexpanded = None;
                }
            }

            // Check unexpanded children (if any priors remain)
            let child_priors = node.get_child_priors();
            let expanded_moves: SmallVec<[u32; 32]> = children.iter().map(|(m, _)| *m).collect();

            for &(move_id, prior) in child_priors.iter() {
                if !expanded_moves.contains(&move_id) {
                    let score = MCTSNode::unexpanded_ucb(prior, parent_visits, c_puct);
                    if score > best_score {
                        best_score = score;
                        best_child_idx = None;
                        best_unexpanded = Some((move_id, prior));
                    }
                }
            }

            match (best_child_idx, best_unexpanded) {
                (Some(idx), _) => {
                    // Apply virtual loss and continue
                    nodes[idx as usize].apply_virtual_loss();
                    current_idx = idx;
                }
                (None, Some((_move_id, _prior))) => {
                    // Need to expand this child - return path for expansion
                    let priors: Vec<f32> = child_priors
                        .iter()
                        .filter(|(m, _)| !expanded_moves.contains(m))
                        .map(|(_, p)| *p)
                        .collect();
                    return Some((path, priors));
                }
                (None, None) => {
                    // No children at all - terminal
                    return Some((path, vec![]));
                }
            }
        }
    }

    /// Expand a leaf node and backpropagate value (Python interface)
    pub fn expand_and_backprop(&self, path: Vec<u32>, children_priors: Vec<(u32, f32)>, value: f32) {
        self.expand_and_backprop_internal(&path, &children_priors, value);
    }

    /// Add a child node to the tree
    #[pyo3(signature = (parent_idx, move_id, prior))]
    pub fn add_child(&self, parent_idx: u32, move_id: u32, prior: f32) -> u32 {
        let child_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let child_node = MCTSNode::new(child_id, move_id, prior, parent_idx);

        let mut nodes = self.nodes.write();
        let child_idx = nodes.len() as u32;
        nodes.push(child_node);

        // Add to parent's children
        let parent = &nodes[parent_idx as usize];
        let mut children = parent.get_children();
        children.push((move_id, child_idx));
        parent.expand(&children);

        child_idx
    }

    /// Get action probabilities based on visit counts
    #[pyo3(signature = (temperature = 1.0))]
    pub fn get_action_probs(&self, temperature: f32) -> Vec<(u32, f32)> {
        let nodes = self.nodes.read();
        let root = &nodes[0];

        if !root.is_expanded() {
            return vec![];
        }

        let children = root.get_children();
        if children.is_empty() {
            // Return priors if no children expanded yet
            return root.get_child_priors().into_iter().collect();
        }

        let mut moves_visits: Vec<(u32, f32)> = children
            .iter()
            .map(|&(move_id, child_idx)| {
                let visits = nodes[child_idx as usize].visit_count() as f32;
                (move_id, visits)
            })
            .collect();

        if moves_visits.is_empty() {
            return vec![];
        }

        // Apply temperature
        if temperature < 0.01 {
            // Greedy - pick best
            let max_visits = moves_visits.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
            for (_, v) in moves_visits.iter_mut() {
                *v = if (*v - max_visits).abs() < 0.001 { 1.0 } else { 0.0 };
            }
        } else {
            // Apply temperature
            for (_, v) in moves_visits.iter_mut() {
                *v = v.powf(1.0 / temperature);
            }
        }

        // Normalize
        let sum: f32 = moves_visits.iter().map(|(_, v)| *v).sum();
        if sum > 0.0 {
            for (_, v) in moves_visits.iter_mut() {
                *v /= sum;
            }
        }

        moves_visits
    }

    /// Get the best move (most visited)
    pub fn best_move(&self) -> Option<u32> {
        let probs = self.get_action_probs(0.0);
        probs.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(m, _)| m)
    }

    /// Get Q-value for root children
    pub fn get_root_q_values(&self) -> Vec<(u32, f32)> {
        let nodes = self.nodes.read();
        let root = &nodes[0];
        let children = root.get_children();

        children
            .iter()
            .map(|&(move_id, child_idx)| {
                let q = nodes[child_idx as usize].q_value();
                (move_id, q)
            })
            .collect()
    }
}
