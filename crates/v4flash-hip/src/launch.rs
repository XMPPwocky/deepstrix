//! [`launch_kernel!`] — a safe-at-the-call-site wrapper around
//! [`Function::launch_raw`](crate::Function::launch_raw).
//!
//! Every kernel launch in `v4flash-kernels` used to hand-marshal its
//! arguments like this:
//!
//! ```ignore
//! let mut a = gate.raw();
//! let mut b = n_rows;
//! let mut args: [*mut c_void; 2] = [
//!     &mut a as *mut _ as *mut c_void,
//!     &mut b as *mut _ as *mut c_void,
//! ];
//! unsafe { function.launch_raw(cfg, stream, &mut args) }
//! ```
//!
//! That triple-cast (`&mut x as *mut _ as *mut c_void`) appeared ~520
//! times — each one a place a type mismatch silently becomes UB. This
//! macro collapses all of them into one audited expansion:
//!
//! ```ignore
//! launch_kernel!(function, cfg, stream, [gate.raw(), n_rows])?;
//! ```
//!
//! # How it stays sound
//!
//! HIP's `kernelParams` ABI is `void**`: an array of pointers, each
//! pointing at one argument *value*. The pointers must be valid for the
//! duration of the `hipModuleLaunchKernel` call (arg bytes are read
//! synchronously during launch, not during async kernel execution — same
//! contract the hand-written version relied on).
//!
//! A `macro_rules!` macro cannot mint a fresh identifier per argument, so
//! we store all arguments in a nested cons-tuple — `(a, (b, (c, ())))` —
//! which gives each value a stable, distinct address. The pointer array
//! is then built by walking that tuple with an accumulating field path
//! (`store.0`, `store.1.0`, `store.1.1.0`, …). The borrows are to
//! disjoint nested fields and are immediately cast to raw pointers, so
//! they never conflict. The cons-tuple is a local that outlives the
//! `launch_raw` call within the macro's block.

/// Launch a kernel with the given config, stream, and argument list.
///
/// Expands to a `Result<(), eyre::Report>` (whatever
/// [`Function::launch_raw`](crate::Function::launch_raw) returns), so use
/// `?` at the call site. The `unsafe` lives inside the expansion.
///
/// ```ignore
/// launch_kernel!(self.function(), cfg, stream, [
///     gate.raw(),
///     up.raw(),
///     n_rows,   // u32 — must match the kernel's C parameter type
///     n_blocks,
/// ])?;
/// ```
///
/// # Safety contract (inherited from `launch_raw`)
///
/// Each argument's Rust type must match the kernel's C signature in size
/// and ABI, in order. The macro guarantees the pointer plumbing is
/// correct; it cannot check that `n_rows: u32` lines up with an `int` vs
/// `unsigned` parameter — that is the caller's responsibility, exactly as
/// it was with the hand-written args array.
#[macro_export]
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $stream:expr, [ $($arg:expr),* $(,)? ]) => {{
        // Stable storage for every argument (see module docs).
        let mut __lk_store = $crate::__launch_cons!($($arg),*);
        // The first RING_ARGS argument words, for the launch ring (`v4flash_hip::ring`).
        let __lk_words_arr = $crate::__launch_words!(@go __lk_store () [] $($arg),*);
        let __lk_words: &[u64] = &__lk_words_arr;
        let __lk_n = if __lk_words.len() < $crate::ring::RING_ARGS { __lk_words.len() } else { $crate::ring::RING_ARGS };
        let mut __lk_args = $crate::__launch_ptrs!(@go __lk_store () [] $($arg),*);
        // SAFETY: every element of `__lk_args` points at the matching field
        // of `__lk_store` (one per kernel param, in declaration order); the
        // store is a live local that outlives this `launch_raw` call. Type/ABI
        // matching of each arg to the kernel signature is the caller's
        // responsibility — the same contract `launch_raw` already documents.
        #[allow(unused_unsafe)]
        unsafe {
            $func.launch_raw_traced($cfg, $stream, &mut __lk_args, &__lk_words[..__lk_n])
        }
    }};
}

/// Build the nested cons-tuple `(a, (b, (c, ())))` that backs the
/// argument pointers. Internal to [`launch_kernel!`].
#[doc(hidden)]
#[macro_export]
macro_rules! __launch_cons {
    () => { () };
    ($head:expr $(, $tail:expr)*) => {
        ($head, $crate::__launch_cons!($($tail),*))
    };
}

/// Walk the cons-tuple stored in `$base`, emitting one
/// `*mut c_void` per argument. `$($pre)*` accumulates the `.1` field
/// path so the Nth argument is reached at `$base .1 .1 … .0`. Internal to
/// [`launch_kernel!`]; the trailing `$arg` list only drives the iteration
/// count — the values come from `$base`.
#[doc(hidden)]
#[macro_export]
macro_rules! __launch_ptrs {
    (@go $base:ident ($($pre:tt)*) [$($acc:tt)*] ) => {
        [ $($acc)* ]
    };
    (@go $base:ident ($($pre:tt)*) [$($acc:tt)*] $head:expr $(, $rest:expr)*) => {
        $crate::__launch_ptrs!(
            @go $base ($($pre)* .1)
            [ $($acc)* &mut $base $($pre)* .0 as *mut _ as *mut ::core::ffi::c_void, ]
            $($rest),*
        )
    };
}

/// As [`__launch_ptrs!`], emitting each argument's ring word
/// ([`crate::ring::arg_word`]) instead of its pointer. Internal to
/// [`launch_kernel!`].
#[doc(hidden)]
#[macro_export]
macro_rules! __launch_words {
    (@go $base:ident ($($pre:tt)*) [$($acc:tt)*] ) => {
        [ $($acc)* ]
    };
    (@go $base:ident ($($pre:tt)*) [$($acc:tt)*] $head:expr $(, $rest:expr)*) => {
        $crate::__launch_words!(
            @go $base ($($pre)* .1)
            [ $($acc)* $crate::ring::arg_word(&$base $($pre)* .0), ]
            $($rest),*
        )
    };
}

#[cfg(test)]
mod tests {
    // Validates the arg-pointer plumbing with no GPU: build the same
    // cons-tuple the macro builds, walk it, and confirm each emitted
    // pointer dereferences back to the original value at the right slot.
    // The `(),(),()` placeholders only drive the walker's iteration count.
    #[test]
    fn arg_ptrs_point_at_their_values() {
        let mut store = crate::__launch_cons!(11u32, 2.5f32, 7i64);
        let args = crate::__launch_ptrs!(@go store () [] (), (), ());

        assert_eq!(args.len(), 3);
        unsafe {
            assert_eq!(*(args[0] as *const u32), 11);
            assert_eq!(*(args[1] as *const f32), 2.5);
            assert_eq!(*(args[2] as *const i64), 7);
        }
    }

    #[test]
    fn arg_words_follow_the_store() {
        let store = crate::__launch_cons!(11u32, 2.5f32, 0x7e00_0000_1000usize as *mut u8);
        let words: [u64; 3] = crate::__launch_words!(@go store () [] (), (), ());
        assert_eq!(words, [11, 2.5f32.to_bits() as u64, 0x7e00_0000_1000]);
    }

    #[test]
    fn empty_arg_list_is_empty_array() {
        let mut _store = crate::__launch_cons!();
        let args: [*mut ::core::ffi::c_void; 0] =
            crate::__launch_ptrs!(@go _store () []);
        assert_eq!(args.len(), 0);
        // As the macro binds it: the element type comes from the slice coercion.
        let words_arr = crate::__launch_words!(@go _store () []);
        let words: &[u64] = &words_arr;
        assert!(words.is_empty());
    }
}
