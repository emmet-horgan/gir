//! Typed value identifiers — SSA-style references to tensor results.
//!
//! Every tensor produced or consumed in the IR graph is referenced via a
//! [`ValueId`].  The [`ValueInfo`] struct pairs a `ValueId` with its static
//! type information (element type and shape).

use std::fmt;

use serde::Serialize;

use crate::dtype::DType;
use crate::shape::Shape;

/// An opaque, unique identifier for a value (tensor) in the IR graph.
///
/// Values are produced by graph inputs or by operation outputs. They are
/// consumed by downstream operations. This is the SSA "register name".
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ValueId(u32);

impl ValueId {
    /// Create a `ValueId` from a raw index.  Normally you should use
    /// [`GraphBuilder`](crate::builder::GraphBuilder) rather than constructing
    /// these directly.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The underlying numeric index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueId(%{})", self.0)
    }
}

/// Static type information associated with a [`ValueId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValueInfo {
    /// Human-readable name (optional, for debugging / serialisation).
    pub name: Option<String>,
    /// Element data type.
    pub dtype: DType,
    /// Tensor shape (may contain symbolic dimensions).
    pub shape: Shape,
}

impl ValueInfo {
    /// Create a new `ValueInfo`.
    #[must_use]
    pub fn new(dtype: DType, shape: Shape) -> Self {
        Self {
            name: None,
            dtype,
            shape,
        }
    }

    /// Attach an optional human-readable name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl fmt::Display for ValueInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{name}: ")?;
        }
        write!(f, "tensor<{}, {}>", self.dtype, self.shape)
    }
}
