use parking_lot::RwLock;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, AtomicI32, Ordering};

/// A node in the MCTS tree, optimized for concurrent access
pub struct MCTSNode {
    /// Unique identifier for this node
    pub id: u32,

    /// Move that led to this node (piece_idx << 24 | orientation << 16 | row << 8 | col)
    pub move_id: u32,

    /// Prior probability from neural network
    pub prior: f32,

    /// Visit count (atomic for thread safety)
    visit_count: AtomicU32,

    /// Value sum (stored as fixed-point for atomic ops: actual = value_sum / 1000000)
    value_sum: AtomicI32,

    /// Virtual loss count (for parallel MCTS)
    virtual_loss: AtomicU32,

    /// Parent node index (u32::MAX if root)
    pub parent: u32,

    /// Children: stored as (move_id, node_index) pairs
    /// Using RwLock for expansion, but reads are lock-free via atomic visit_count
    children: RwLock<SmallVec<[(u32, u32); 32]>>,

    /// Child priors (stored until expanded)
    child_priors: RwLock<SmallVec<[(u32, f32); 64]>>,

    /// Whether this node has been expanded
    is_expanded: std::sync::atomic::AtomicBool,
}

impl MCTSNode {
    pub fn new(id: u32, move_id: u32, prior: f32, parent: u32) -> Self {
        Self {
            id,
            move_id,
            prior,
            visit_count: AtomicU32::new(0),
            value_sum: AtomicI32::new(0),
            virtual_loss: AtomicU32::new(0),
            parent,
            children: RwLock::new(SmallVec::new()),
            child_priors: RwLock::new(SmallVec::new()),
            is_expanded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn visit_count(&self) -> u32 {
        self.visit_count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn value_sum(&self) -> f32 {
        self.value_sum.load(Ordering::Relaxed) as f32 / 1_000_000.0
    }

    #[inline]
    pub fn q_value(&self) -> f32 {
        let visits = self.visit_count();
        if visits == 0 {
            0.0
        } else {
            self.value_sum() / visits as f32
        }
    }

    #[inline]
    pub fn virtual_loss_count(&self) -> u32 {
        self.virtual_loss.load(Ordering::Relaxed)
    }

    /// Calculate UCB score for child selection
    #[inline]
    pub fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let visits = self.visit_count() + self.virtual_loss_count();
        let q = if visits == 0 {
            0.0
        } else {
            // Adjust Q for virtual loss
            let vl = self.virtual_loss_count() as f32;
            (self.value_sum() - vl * 3.0) / visits as f32
        };

        let exploration = c_puct * self.prior * (parent_visits as f32).sqrt() / (1.0 + visits as f32);
        q + exploration
    }

    /// Calculate UCB score for an unexpanded child (no node yet)
    #[inline]
    pub fn unexpanded_ucb(prior: f32, parent_visits: u32, c_puct: f32) -> f32 {
        c_puct * prior * (parent_visits as f32).sqrt()
    }

    /// Apply virtual loss (for parallel MCTS)
    #[inline]
    pub fn apply_virtual_loss(&self) {
        self.visit_count.fetch_add(1, Ordering::Relaxed);
        self.virtual_loss.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove virtual loss
    #[inline]
    pub fn remove_virtual_loss(&self) {
        self.visit_count.fetch_sub(1, Ordering::Relaxed);
        self.virtual_loss.fetch_sub(1, Ordering::Relaxed);
    }

    /// Backpropagate a value
    #[inline]
    pub fn backprop(&self, value: f32) {
        self.visit_count.fetch_add(1, Ordering::Relaxed);
        // Convert to fixed-point and add atomically
        let value_fixed = (value * 1_000_000.0) as i32;
        self.value_sum.fetch_add(value_fixed, Ordering::Relaxed);
    }

    /// Check if expanded
    #[inline]
    pub fn is_expanded(&self) -> bool {
        self.is_expanded.load(Ordering::Acquire)
    }

    /// Set child priors (before expansion)
    pub fn set_child_priors(&self, priors: &[(u32, f32)]) {
        let mut cp = self.child_priors.write();
        cp.clear();
        cp.extend(priors.iter().copied());
    }

    /// Get child priors
    pub fn get_child_priors(&self) -> SmallVec<[(u32, f32); 64]> {
        self.child_priors.read().clone()
    }

    /// Mark as expanded and add children
    pub fn expand(&self, children_ids: &[(u32, u32)]) {
        let mut children = self.children.write();
        children.clear();
        children.extend(children_ids.iter().copied());
        self.is_expanded.store(true, Ordering::Release);
    }

    /// Get children (returns copy to avoid holding lock)
    pub fn get_children(&self) -> SmallVec<[(u32, u32); 32]> {
        self.children.read().clone()
    }

    /// Check if has children
    pub fn has_children(&self) -> bool {
        !self.children.read().is_empty()
    }
}
