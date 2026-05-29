//! Hash-router selection for the bootstrap layers (L < N_HASH_LAYERS).
//!
//! Mirrors `layer_hash_selected_experts` + `layer_hash_router_weights_one`
//! (ds4.c:5209, 5260). The learned router (L ≥ N_HASH_LAYERS) is a
//! straightforward matvec + topk and lives inline in the forward path.

use crate::config::{EXPERT_WEIGHT_SCALE, N_EXPERT_USED};

fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

/// Hash-router selection from `tid2eid[token_id * 6 + slot]`. Returns
/// `(selected[6], weights[6])`.
pub fn hash_router_select(
    tid2eid: &[i32],
    token_id: i32,
    logits_host: &[f32],
) -> ([i32; 6], [f32; 6]) {
    let mut selected = [0i32; N_EXPERT_USED];
    for i in 0..N_EXPERT_USED {
        selected[i] = tid2eid[(token_id as usize) * N_EXPERT_USED + i];
    }
    let mut w = [0f32; N_EXPERT_USED];
    let mut sum = 0f32;
    for i in 0..N_EXPERT_USED {
        let p = softplus_stable(logits_host[selected[i] as usize]).sqrt();
        w[i] = p;
        sum += p;
    }
    if sum < 6.103515625e-5 {
        sum = 6.103515625e-5;
    }
    for i in 0..N_EXPERT_USED {
        w[i] = w[i] / sum * EXPERT_WEIGHT_SCALE;
    }
    (selected, w)
}
