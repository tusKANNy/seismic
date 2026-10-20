//! The forward-index representations a Seismic index can be built with, by their command-line
//! names (`--value-type` in `build_inverted_index` and `perf_inverted_index`).
//!
//! [`with_dataset_type!`](crate::with_dataset_type) maps a `(component type, value type)` pair
//! to the dataset type, so that the binaries and the experiment scripts agree on one list.
//!
//! | Value type | Component types | Dataset |
//! |---|---|---|
//! | `f32` | `u16`, `u32` | plain sparse, `f32` values |
//! | `f16`, `bf16` | `u16`, `u32` | plain sparse, values cast to `f16` / `bf16` |
//! | `fixedu8`, `fixedu16` | `u16`, `u32` | plain sparse, fixed-point values (one scale for all components) |
//! | `scalaru8`, `scalaru16` | `u16`, `u32` | plain sparse, 8/16-bit codes with one trained step per component |
//! | `blocked-sparse` | `u16` | 64-byte blocks, `f16` values |
//! | `blocked-sparse-scalaru8`, `blocked-sparse-scalaru16` | `u16` | blocks, 8/16-bit codes with one step per component |
//! | `dotpacking8-fixedu8` | `u16` | DotPacking8, fixed-point values |
//! | `dotpacking8-scalaru8`, `dotpacking8-scalaru16` | `u16`, `u32` (bridged) | DotPacking8, 8/16-bit codes with one step per component |
//! | `dotpacking8-f16` | `u16` | DotPacking8, `f16` values |
//! | `dotpacking8-scalar4bit` | `u16` | DotPacking8, 4-bit codes with one step per component |
//!
//! `dotpacking8-scalar` is accepted as the name `dotpacking8-scalaru8` had before.

use half::{bf16, f16};
use vectorium::{
    BlockedSparseScalarU8Dataset, BlockedSparseScalarU16Dataset,
    BridgedDotPacking8U32ScalarU8Dataset, BridgedDotPacking8U32ScalarU16Dataset,
    DotPacking8U16F16Dataset, DotPacking8U16FixedU8Dataset, DotPacking8U16Scalar4BitDataset,
    DotPacking8U16ScalarU8Dataset, DotPacking8U16ScalarU16Dataset, DotProduct, FixedU8Q, FixedU16Q,
    PlainSparseDataset, ScalarSparseDataset, ScalarU8SparseDataset, ScalarU16SparseDataset,
};

pub type F32Dataset<C> = PlainSparseDataset<C, f32, DotProduct>;
pub type F16Dataset<C> = ScalarSparseDataset<C, f32, f16, DotProduct>;
pub type Bf16Dataset<C> = ScalarSparseDataset<C, f32, bf16, DotProduct>;
pub type FixedU8Dataset<C> = ScalarSparseDataset<C, f32, FixedU8Q, DotProduct>;
pub type FixedU16Dataset<C> = ScalarSparseDataset<C, f32, FixedU16Q, DotProduct>;
pub type ScalarU8Dataset<C> = ScalarU8SparseDataset<C, DotProduct>;
pub type ScalarU16Dataset<C> = ScalarU16SparseDataset<C, DotProduct>;
pub use vectorium::BlockedSparseDataset;
pub type BlockedScalarU8Dataset = BlockedSparseScalarU8Dataset;
pub type BlockedScalarU16Dataset = BlockedSparseScalarU16Dataset;
pub type DotPacking8FixedU8Dataset = DotPacking8U16FixedU8Dataset;
pub type DotPacking8ScalarU8Dataset = DotPacking8U16ScalarU8Dataset;
pub type DotPacking8BridgedScalarU8Dataset = BridgedDotPacking8U32ScalarU8Dataset;
pub type DotPacking8ScalarU16Dataset = DotPacking8U16ScalarU16Dataset;
pub type DotPacking8BridgedScalarU16Dataset = BridgedDotPacking8U32ScalarU16Dataset;
pub type DotPacking8F16Dataset = DotPacking8U16F16Dataset;
pub type DotPacking8Scalar4BitDataset = DotPacking8U16Scalar4BitDataset;

/// Every value type accepted by [`with_dataset_type!`](crate::with_dataset_type).
pub const VALUE_TYPES: &[&str] = &[
    "f32",
    "f16",
    "bf16",
    "fixedu8",
    "fixedu16",
    "scalaru8",
    "scalaru16",
    "blocked-sparse",
    "blocked-sparse-scalaru8",
    "blocked-sparse-scalaru16",
    "dotpacking8-fixedu8",
    "dotpacking8-scalaru8",
    "dotpacking8-scalaru16",
    "dotpacking8-f16",
    "dotpacking8-scalar4bit",
];

/// For a value type that existed before vectorium 0.11 and was removed, what to use instead.
pub fn removed_value_type_hint(value_type: &str) -> Option<&'static str> {
    match value_type {
        "dotvbyte" | "dotvbyte-scalar" | "dotvbyte-optimistic" | "dotvbyte-optimistic-scalar" => {
            Some(
                "DotVByte was removed in vectorium 0.11; use 'dotpacking8-scalaru8' (u16 or u32 components)",
            )
        }
        "dotpacking8-optimistic-scalar" => Some(
            "use 'dotpacking8-scalaru8' with component type 'u32' (the bridged u32 DotPacking8)",
        ),
        "dotpacking8-sparse-scalar"
        | "dotpacking8-centroid4bit"
        | "dotpackingsparse-f16"
        | "packed-centroid"
        | "packed-variable-uniform" => Some(
            "this research-only encoder was removed in vectorium 0.11; see VALUE_TYPES for the available ones",
        ),
        _ => None,
    }
}

/// Error message for a `(component type, value type)` pair that `with_dataset_type!` rejects.
pub fn unsupported_message(component_type: &str, value_type: &str) -> String {
    let mut msg = format!(
        "unsupported combination: component type '{component_type}', value type '{value_type}'."
    );
    if let Some(hint) = removed_value_type_hint(value_type) {
        msg.push_str(&format!(" '{value_type}': {hint}."));
    } else if VALUE_TYPES.contains(&value_type) {
        msg.push_str(" This value type does not support this component type.");
    } else {
        msg.push_str(&format!(" Value types: {}.", VALUE_TYPES.join(", ")));
    }
    msg
}

/// Expands `$callback!(C, DatasetType, route)` for the dataset type selected by a component
/// type name (`"u16"`, `"u32"`) and a value type name (see [`VALUE_TYPES`](crate::value_types::VALUE_TYPES)).
/// Evaluates to `Ok(result of the callback)`, or to `Err(message)` for an unsupported pair.
///
/// `route` tells how the index is built, so that a build binary can match on it:
/// - `plain`: build directly on the `f32` dataset;
/// - `quantize_first`: convert the dataset, then build on the converted values;
/// - `convert_after`: build on the `f32` dataset, then convert the forward index. Used by every
///   encoder whose stored values are not comparable across components (per-component steps,
///   packed layouts), as pruning and clustering need the real values.
#[macro_export]
macro_rules! with_dataset_type {
    ($component:expr, $value:expr, $callback:ident) => {{
        use $crate::value_types as vt;
        match ($component, $value) {
            ("u16", "f32") => Ok($callback!(u16, vt::F32Dataset<u16>, plain)),
            ("u32", "f32") => Ok($callback!(u32, vt::F32Dataset<u32>, plain)),
            ("u16", "f16") => Ok($callback!(u16, vt::F16Dataset<u16>, quantize_first)),
            ("u32", "f16") => Ok($callback!(u32, vt::F16Dataset<u32>, quantize_first)),
            ("u16", "bf16") => Ok($callback!(u16, vt::Bf16Dataset<u16>, quantize_first)),
            ("u32", "bf16") => Ok($callback!(u32, vt::Bf16Dataset<u32>, quantize_first)),
            ("u16", "fixedu16") => Ok($callback!(u16, vt::FixedU16Dataset<u16>, quantize_first)),
            ("u32", "fixedu16") => Ok($callback!(u32, vt::FixedU16Dataset<u32>, quantize_first)),
            ("u16", "fixedu8") => Ok($callback!(u16, vt::FixedU8Dataset<u16>, convert_after)),
            ("u32", "fixedu8") => Ok($callback!(u32, vt::FixedU8Dataset<u32>, convert_after)),
            ("u16", "scalaru8") => Ok($callback!(u16, vt::ScalarU8Dataset<u16>, convert_after)),
            ("u32", "scalaru8") => Ok($callback!(u32, vt::ScalarU8Dataset<u32>, convert_after)),
            ("u16", "scalaru16") => Ok($callback!(u16, vt::ScalarU16Dataset<u16>, convert_after)),
            ("u32", "scalaru16") => Ok($callback!(u32, vt::ScalarU16Dataset<u32>, convert_after)),
            ("u16", "blocked-sparse") => {
                Ok($callback!(u16, vt::BlockedSparseDataset, convert_after))
            }
            ("u16", "blocked-sparse-scalaru8") => {
                Ok($callback!(u16, vt::BlockedScalarU8Dataset, convert_after))
            }
            ("u16", "blocked-sparse-scalaru16") => {
                Ok($callback!(u16, vt::BlockedScalarU16Dataset, convert_after))
            }
            ("u16", "dotpacking8-fixedu8") => Ok($callback!(
                u16,
                vt::DotPacking8FixedU8Dataset,
                convert_after
            )),
            ("u16", "dotpacking8-scalaru8" | "dotpacking8-scalar") => Ok($callback!(
                u16,
                vt::DotPacking8ScalarU8Dataset,
                convert_after
            )),
            ("u32", "dotpacking8-scalaru8" | "dotpacking8-scalar") => Ok($callback!(
                u32,
                vt::DotPacking8BridgedScalarU8Dataset,
                convert_after
            )),
            ("u16", "dotpacking8-scalaru16") => Ok($callback!(
                u16,
                vt::DotPacking8ScalarU16Dataset,
                convert_after
            )),
            ("u32", "dotpacking8-scalaru16") => Ok($callback!(
                u32,
                vt::DotPacking8BridgedScalarU16Dataset,
                convert_after
            )),
            ("u16", "dotpacking8-f16") => {
                Ok($callback!(u16, vt::DotPacking8F16Dataset, convert_after))
            }
            ("u16", "dotpacking8-scalar4bit") => Ok($callback!(
                u16,
                vt::DotPacking8Scalar4BitDataset,
                convert_after
            )),
            (component, value) => Err(vt::unsupported_message(component, value)),
        }
    }};
}
