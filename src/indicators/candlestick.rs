use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap;
use tulip_rs::candle_indicators::types::ForecastType as RustForecastType;
use tulip_rs::indicators::candlestick as rust_cdl;

// ---------------------------------------------------------------------------
// ForecastType — exposed as a Python class with class-level constants
// ---------------------------------------------------------------------------

/// Filter enum for candlestick pattern detection.
///
/// Pass one of these values as the `forecast_type` argument to narrow the
/// patterns that are returned to a specific signal direction.
///
/// Example:
///     >>> from tulip_rs.indicators.candlestick import ForecastType
///     >>> patterns, state = candlestick(o, h, l, c, forecast_type=ForecastType.BullishReversal)
#[pyclass(from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ForecastType {
    inner: RustForecastType,
}

#[pymethods]
impl ForecastType {
    /// BearishReversal — pattern signals a reversal from uptrend to downtrend
    #[classattr]
    #[allow(non_snake_case)]
    fn BearishReversal() -> Self {
        Self {
            inner: RustForecastType::BearishReversal,
        }
    }

    /// BullishReversal — pattern signals a reversal from downtrend to uptrend
    #[classattr]
    #[allow(non_snake_case)]
    fn BullishReversal() -> Self {
        Self {
            inner: RustForecastType::BullishReversal,
        }
    }

    /// BearishContinuation — pattern signals continuation of downtrend
    #[classattr]
    #[allow(non_snake_case)]
    fn BearishContinuation() -> Self {
        Self {
            inner: RustForecastType::BearishContinuation,
        }
    }

    /// BullishContinuation — pattern signals continuation of uptrend
    #[classattr]
    #[allow(non_snake_case)]
    fn BullishContinuation() -> Self {
        Self {
            inner: RustForecastType::BullishContinuation,
        }
    }

    /// BearishReversalOrContinuation — bearish signal, either direction
    #[classattr]
    #[allow(non_snake_case)]
    fn BearishReversalOrContinuation() -> Self {
        Self {
            inner: RustForecastType::BearishReversalOrContinuation,
        }
    }

    /// BullishReversalOrContinuation — bullish signal, either direction
    #[classattr]
    #[allow(non_snake_case)]
    fn BullishReversalOrContinuation() -> Self {
        Self {
            inner: RustForecastType::BullishReversalOrContinuation,
        }
    }

    fn __repr__(&self) -> String {
        format!("ForecastType.{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        match self.inner {
            RustForecastType::BearishReversal => 0,
            RustForecastType::BullishReversal => 1,
            RustForecastType::BearishContinuation => 2,
            RustForecastType::BullishContinuation => 3,
            RustForecastType::BearishReversalOrContinuation => 4,
            RustForecastType::BullishReversalOrContinuation => 5,
        }
    }

    /// Return the string name of this variant
    #[getter]
    fn value(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// CandlestickState — streaming state wrapper
// ---------------------------------------------------------------------------

/// Streaming state for the candlestick indicator.
///
/// Returned by `candlestick()` and used to continue processing new bars
/// without re-processing historical data.
///
/// Methods:
///     batch_indicator(open, high, low, close, forecast_type=None) -> list
///     to_json() -> str
///     from_json(json_str) -> CandlestickState
#[pyclass]
pub struct CandlestickState {
    inner: Box<rust_cdl::IndicatorState>,
}

#[pymethods]
impl CandlestickState {
    /// Process new bars and return detected patterns for each new bar.
    ///
    /// Args:
    ///     open:          list[float] or numpy array of open prices
    ///     high:          list[float] or numpy array of high prices
    ///     low:           list[float] or numpy array of low prices
    ///     close:         list[float] or numpy array of close prices
    ///     forecast_type: optional ForecastType filter (default: all patterns)
    ///
    /// Returns:
    ///     list[list[str] | None] — one entry per new bar:
    ///         None           → no pattern detected
    ///         list[str]      → one or more pattern names detected
    #[pyo3(signature = (open, high, low, close, forecast_type=None))]
    fn batch_indicator(
        &mut self,
        open: PyReadonlyArray1<f64>,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        forecast_type: Option<ForecastType>,
    ) -> PyResult<Vec<Option<Vec<HashMap<String, String>>>>> {
        let ft = forecast_type.map(|f| f.inner);
        let inputs: [&[f64]; 4] = [
            open.as_slice()?,
            high.as_slice()?,
            low.as_slice()?,
            close.as_slice()?,
        ];

        match self.inner.batch_indicator(&inputs, ft) {
            Ok(output) => Ok(convert_output(output)),
            Err(e) => Err(map_error(e)),
        }
    }

    /// Serialise the state to a JSON string for persistence.
    ///
    /// Example:
    ///     >>> json_str = state.to_json()
    ///     >>> with open("state.json", "w") as f:
    ///     ...     f.write(json_str)
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&*self.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {}", e))
        })
    }

    /// Restore a CandlestickState from a JSON string previously produced by `to_json()`.
    ///
    /// Example:
    ///     >>> state = CandlestickState.from_json(json_str)
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, json_str: &str) -> PyResult<Self> {
        let inner: rust_cdl::IndicatorState = serde_json::from_str(json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid or corrupted indicator state: {}",
                e
            ))
        })?;
        Ok(CandlestickState {
            inner: Box::new(inner),
        })
    }

    fn __repr__(&self) -> String {
        "CandlestickState(internal)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// One-shot batch candlestick pattern detection.
///
/// Processes a complete OHLC dataset and returns all detected patterns plus
/// a streaming state that can be used to continue with new bars.
///
/// Args:
///     open:          list[float] or numpy array of open prices
///     high:          list[float] or numpy array of high prices
///     low:           list[float] or numpy array of low prices
///     close:         list[float] or numpy array of close prices
///     options:       [candle_period, trend_period, trend_signal_period] (default [14, 20, 9])
///     forecast_type: optional ForecastType filter (default: all patterns for current trend)
///
/// Returns:
///     (patterns, state) where:
///       patterns — list[list[str] | None], one entry per output bar
///       state    — CandlestickState for streaming continuation
///
/// Example:
///     >>> patterns, state = candlestick(o, h, l, c, options=[14, 20, 9])
///     >>> for i, p in enumerate(patterns):
///     ...     if p:
///     ...         print(f"Bar {i}: {p}")
#[pyfunction]
#[pyo3(signature = (open, high, low, close, options=None, forecast_type=None))]
pub fn candlestick(
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    options: Option<Vec<f64>>,
    forecast_type: Option<ForecastType>,
) -> PyResult<(Vec<Option<Vec<HashMap<String, String>>>>, CandlestickState)> {
    let opts = resolve_options(options)?;
    let ft = forecast_type.map(|f| f.inner);
    let inputs: [&[f64]; 4] = [
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
    ];

    match rust_cdl::CandleStick::indicator(&inputs, &opts, ft) {
        Ok((output, state)) => Ok((
            convert_output(output),
            CandlestickState {
                inner: Box::new(state),
            },
        )),
        Err(e) => Err(map_error(e)),
    }
}

/// Return the minimum number of bars required before any output is produced.
///
/// Args:
///     options: [candle_period, trend_period, trend_signal_period] (default [14, 20, 9])
///
/// Returns:
///     int — minimum bar count
///
/// Example:
///     >>> min_bars = min_data([14, 20, 9])
///     >>> if len(close) < min_bars:
///     ...     raise ValueError("Not enough data")
#[pyfunction]
#[pyo3(signature = (options=None))]
pub fn min_data(options: Option<Vec<f64>>) -> PyResult<usize> {
    let opts = resolve_options(options)?;
    Ok(rust_cdl::CandleStick::min_data(&opts))
}

/// Return indicator metadata as a dict.
///
/// Returns:
///     dict with keys: name, full_name, inputs, options, outputs
///
/// Example:
///     >>> i = info()
///     >>> print(i["full_name"])
///     Candle Stick Indicator
#[pyfunction]
pub fn info(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    crate::utils::info_to_pydict(py, rust_cdl::CandleStick::INFO)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register_candlestick_module(
    parent_module: &pyo3::Bound<'_, PyModule>,
) -> pyo3::PyResult<()> {
    let submodule = PyModule::new(parent_module.py(), "candlestick")?;

    submodule.add_function(pyo3::wrap_pyfunction!(candlestick, &submodule)?)?;
    submodule.add_function(pyo3::wrap_pyfunction!(min_data, &submodule)?)?;
    submodule.add_function(pyo3::wrap_pyfunction!(info, &submodule)?)?;

    submodule.add_class::<CandlestickState>()?;
    submodule.add_class::<ForecastType>()?;

    parent_module.add_submodule(&submodule)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse `options` (defaulting to [14, 20, 9]) and validate count.
fn resolve_options(options: Option<Vec<f64>>) -> PyResult<[f64; rust_cdl::OPTIONS]> {
    let v = options.unwrap_or_else(|| vec![14.0, 20.0, 9.0]);
    if v.len() != rust_cdl::OPTIONS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "options must have exactly {} values [candle_period, trend_period, trend_signal_period], got {}",
            rust_cdl::OPTIONS,
            v.len()
        )));
    }
    Ok([v[0], v[1], v[2]])
}

/// Convert `Vec<Option<Vec<CandlePattern>>>` → `Vec<Option<Vec<HashMap<String,String>>>>`.
/// Each pattern becomes a dict with name, full_name, japanese_name, bars, forecast.
fn convert_output(
    raw: Vec<Option<Vec<tulip_rs::candle_indicators::candle_patterns::CandlePattern>>>,
) -> Vec<Option<Vec<HashMap<String, String>>>> {
    raw.into_iter()
        .map(|entry| {
            entry.map(|patterns| {
                patterns
                    .into_iter()
                    .map(|p| {
                        let info = p.get_info();
                        let mut m = HashMap::new();
                        m.insert("name".to_string(), format!("{:?}", p));
                        m.insert("full_name".to_string(), info.full_name.to_string());
                        m.insert("japanese_name".to_string(), info.japanese_name.to_string());
                        m.insert("bars".to_string(), info.bars.to_string());
                        m.insert("forecast".to_string(), format!("{:?}", info.forecast));
                        m
                    })
                    .collect()
            })
        })
        .collect()
}

/// Map a Rust `IndicatorError` to a Python `ValueError`.
fn map_error(e: tulip_rs::types::IndicatorError) -> PyErr {
    use tulip_rs::types::IndicatorError;
    let msg = match e {
        IndicatorError::NotEnoughData => {
            "Not enough data: supply at least min_data(options) bars".to_string()
        }
        IndicatorError::InvalidInputs => {
            "All input arrays (open, high, low, close) must have the same length".to_string()
        }
        IndicatorError::InvalidOptions => {
            "All options must be >= 1 (candle_period, trend_period, trend_signal_period)"
                .to_string()
        }
        IndicatorError::InvalidIndicatorState => "Invalid or corrupted indicator state".to_string(),
    };
    pyo3::exceptions::PyValueError::new_err(msg)
}
