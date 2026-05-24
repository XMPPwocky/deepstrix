//! HIP graph wrappers (M14).
//!
//! Two usage patterns are supported:
//!
//! * **Capture mode** — call [`Stream::begin_capture`] /
//!   [`Stream::end_capture`] around the existing kernel-launch calls.
//!   HIP records them into a [`Graph`] which can be instantiated as a
//!   [`GraphExec`] and launched as a single API call. After
//!   instantiation, per-token-varying kernel args can be patched in
//!   place via [`GraphExec::set_kernel_node_params`] using nodes pulled
//!   out of the original graph via [`Graph::nodes`].
//! * **Explicit construction** — call [`Graph::new`] then add nodes
//!   manually. Not used yet by the orchestrator.
//!
//! Capture mode is the much smaller refactor for an existing
//! kernel-launch codebase, so the orchestrator uses it.

use std::ptr;

use color_eyre::eyre;

use crate::error::check_eyre;
use crate::stream::Stream;
use crate::sys;

/// A captured (or hand-built) HIP graph. Cheap; just a topology
/// descriptor. To execute, instantiate into a [`GraphExec`].
pub struct Graph {
    raw: sys::hipGraph_t,
}

// SAFETY: HIP graph handles are reference-counted on the runtime side and
// safe to move/share across threads as long as no two threads call
// hipGraph* on the same handle concurrently. We don't.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    /// Build an empty graph.
    pub fn new() -> eyre::Result<Self> {
        let mut raw: sys::hipGraph_t = ptr::null_mut();
        check_eyre(unsafe { sys::hipGraphCreate(&mut raw, 0) }, "hipGraphCreate")?;
        Ok(Self { raw })
    }

    /// Take ownership of a raw HIP graph handle (e.g. the one returned
    /// from `hipStreamEndCapture`).
    pub fn from_raw(raw: sys::hipGraph_t) -> Self {
        Self { raw }
    }

    pub fn raw(&self) -> sys::hipGraph_t {
        self.raw
    }

    /// Enumerate all nodes in capture order. Useful for identifying the
    /// kernel nodes you want to update with
    /// [`GraphExec::set_kernel_node_params`].
    pub fn nodes(&self) -> eyre::Result<Vec<sys::hipGraphNode_t>> {
        let mut count: usize = 0;
        check_eyre(
            unsafe { sys::hipGraphGetNodes(self.raw, ptr::null_mut(), &mut count) },
            "hipGraphGetNodes (count)",
        )?;
        let mut nodes = vec![ptr::null_mut(); count];
        let mut count2 = count;
        check_eyre(
            unsafe { sys::hipGraphGetNodes(self.raw, nodes.as_mut_ptr(), &mut count2) },
            "hipGraphGetNodes (fill)",
        )?;
        nodes.truncate(count2);
        Ok(nodes)
    }

    /// Instantiate the graph for execution. The returned [`GraphExec`]
    /// can be launched repeatedly; nodes can be patched in-place.
    pub fn instantiate(&self) -> eyre::Result<GraphExec> {
        let mut exec: sys::hipGraphExec_t = ptr::null_mut();
        let mut log = [0i8; 1024];
        let mut error_node: sys::hipGraphNode_t = ptr::null_mut();
        let rc = unsafe {
            sys::hipGraphInstantiate(
                &mut exec,
                self.raw,
                &mut error_node,
                log.as_mut_ptr(),
                log.len(),
            )
        };
        if rc != sys::HIP_SUCCESS {
            // Extract any HIP log string for diagnostics.
            let log_str = unsafe { std::ffi::CStr::from_ptr(log.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(color_eyre::eyre::eyre!(
                "hipGraphInstantiate failed (rc={rc}, error_node={:?}, log={:?})",
                error_node,
                log_str
            ));
        }
        Ok(GraphExec { raw: exec })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let rc = unsafe { sys::hipGraphDestroy(self.raw) };
            if rc != sys::HIP_SUCCESS {
                tracing::warn!(code = rc, "hipGraphDestroy failed during drop");
            }
        }
    }
}

/// An instantiated graph ready to launch. Owns device-side resources.
pub struct GraphExec {
    raw: sys::hipGraphExec_t,
}

// SAFETY: same reasoning as Graph — we serialize all use behind &mut
// borrows or external locks.
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    pub fn raw(&self) -> sys::hipGraphExec_t {
        self.raw
    }

    /// Launch the graph on `stream`. Single async API call irrespective
    /// of how many nodes the graph contains — this is the whole point.
    pub fn launch(&self, stream: &Stream) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipGraphLaunch(self.raw, stream.raw()) },
            "hipGraphLaunch",
        )
    }

    /// Update the kernel-launch parameters of one node in the
    /// instantiated graph. Caller is responsible for keeping the
    /// `kernel_params` pointer array alive for the duration of the next
    /// launch — HIP captures the *pointers*, not the data.
    ///
    /// # Safety
    /// `node` must be a kernel node that exists in the graph this
    /// `GraphExec` was instantiated from. `params` must be valid for
    /// the kernel ABI.
    pub unsafe fn set_kernel_node_params(
        &self,
        node: sys::hipGraphNode_t,
        params: &sys::hipKernelNodeParams,
    ) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipGraphExecKernelNodeSetParams(self.raw, node, params) },
            "hipGraphExecKernelNodeSetParams",
        )
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let rc = unsafe { sys::hipGraphExecDestroy(self.raw) };
            if rc != sys::HIP_SUCCESS {
                tracing::warn!(code = rc, "hipGraphExecDestroy failed during drop");
            }
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new().expect("hipGraphCreate failed")
    }
}

/// Fetch a kernel node's current params (useful as a "scaffold" you
/// then mutate and pass back via `set_kernel_node_params`). Returns the
/// raw struct; caller is responsible for any pointer fields it
/// contains.
///
/// # Safety
/// `node` must be a kernel node.
pub unsafe fn kernel_node_params(
    node: sys::hipGraphNode_t,
) -> eyre::Result<sys::hipKernelNodeParams> {
    let mut p = sys::hipKernelNodeParams {
        blockDim: sys::hipDim3::default(),
        extra: ptr::null_mut(),
        func: ptr::null_mut(),
        gridDim: sys::hipDim3::default(),
        kernelParams: ptr::null_mut(),
        sharedMemBytes: 0,
    };
    check_eyre(
        unsafe { sys::hipGraphKernelNodeGetParams(node, &mut p) },
        "hipGraphKernelNodeGetParams",
    )?;
    Ok(p)
}
