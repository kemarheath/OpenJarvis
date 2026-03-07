//! PyO3 bindings for learning/router policy types and optimization.

use openjarvis_learning::RouterPolicy;
use pyo3::prelude::*;

#[pyclass(name = "HeuristicRouter")]
pub struct PyHeuristicRouter {
    inner: openjarvis_learning::HeuristicRouter,
}

#[pymethods]
impl PyHeuristicRouter {
    #[new]
    #[pyo3(signature = (default_model="qwen3:8b", code_model=None, math_model=None, fast_model=None))]
    fn new(
        default_model: &str,
        code_model: Option<String>,
        math_model: Option<String>,
        fast_model: Option<String>,
    ) -> Self {
        Self {
            inner: openjarvis_learning::HeuristicRouter::new(
                default_model.to_string(),
                code_model,
                math_model,
                fast_model,
            ),
        }
    }

    fn select_model(&self, query: &str, has_code: bool, has_math: bool) -> String {
        let ctx = openjarvis_core::RoutingContext {
            query: query.to_string(),
            query_length: query.len(),
            has_code,
            has_math,
            ..Default::default()
        };
        self.inner.select_model(&ctx)
    }
}

#[pyclass(name = "BanditRouterPolicy")]
pub struct PyBanditRouterPolicy {
    inner: openjarvis_learning::BanditRouterPolicy,
}

#[pymethods]
impl PyBanditRouterPolicy {
    #[new]
    #[pyo3(signature = (models, strategy="thompson"))]
    fn new(models: Vec<String>, strategy: &str) -> Self {
        let strat = match strategy {
            "ucb1" | "UCB1" => openjarvis_learning::bandit::BanditStrategy::UCB1,
            _ => openjarvis_learning::bandit::BanditStrategy::ThompsonSampling,
        };
        Self {
            inner: openjarvis_learning::BanditRouterPolicy::new(models, strat),
        }
    }

    fn select_model(&self) -> String {
        let ctx = openjarvis_core::RoutingContext::default();
        self.inner.select_model(&ctx)
    }

    fn update(&self, model: &str, reward: f64) {
        self.inner.update(model, reward);
    }
}

#[pyclass(name = "GRPORouterPolicy")]
pub struct PyGRPORouterPolicy {
    inner: openjarvis_learning::GRPORouterPolicy,
}

#[pymethods]
impl PyGRPORouterPolicy {
    #[new]
    #[pyo3(signature = (models, temperature=1.0))]
    fn new(models: Vec<String>, temperature: f64) -> Self {
        Self {
            inner: openjarvis_learning::GRPORouterPolicy::new(models, temperature),
        }
    }

    fn select_model(&self) -> String {
        let ctx = openjarvis_core::RoutingContext::default();
        self.inner.select_model(&ctx)
    }

    fn update_weights(&self, rewards_json: &str) -> PyResult<()> {
        let rewards: Vec<(String, f64)> = serde_json::from_str(rewards_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.inner.update_weights(&rewards);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Optimization store
// ---------------------------------------------------------------------------

#[pyclass(name = "OptimizationStore")]
pub struct PyOptimizationStore {
    inner: parking_lot::Mutex<openjarvis_learning::optimize::OptimizationStore>,
}

#[pymethods]
impl PyOptimizationStore {
    /// Open or create a store at the given database path.
    /// Use `":memory:"` for an in-memory store.
    #[new]
    #[pyo3(signature = (path=":memory:"))]
    fn new(path: &str) -> PyResult<Self> {
        let store = if path == ":memory:" {
            openjarvis_learning::optimize::OptimizationStore::in_memory()
        } else {
            openjarvis_learning::optimize::OptimizationStore::open(path)
        };
        let store =
            store.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self {
            inner: parking_lot::Mutex::new(store),
        })
    }

    /// Save a trial result (JSON string) for a given run_id.
    fn save_trial(&self, run_id: &str, trial_json: &str) -> PyResult<()> {
        let trial: openjarvis_learning::optimize::TrialResult =
            serde_json::from_str(trial_json)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.inner
            .lock()
            .save_trial(run_id, &trial)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Retrieve an optimization run by id. Returns JSON string or None.
    fn get_run(&self, run_id: &str) -> PyResult<Option<String>> {
        let run = self
            .inner
            .lock()
            .get_run(run_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        match run {
            Some(r) => Ok(Some(
                serde_json::to_string(&r).unwrap_or_else(|_| "{}".into()),
            )),
            None => Ok(None),
        }
    }

    /// List recent optimization runs. Returns JSON string.
    #[pyo3(signature = (limit=50))]
    fn list_runs(&self, limit: usize) -> PyResult<String> {
        let summaries = self
            .inner
            .lock()
            .list_runs(limit)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        // RunSummary doesn't derive Serialize, so manually build JSON.
        let items: Vec<serde_json::Value> = summaries
            .iter()
            .map(|s| {
                serde_json::json!({
                    "run_id": s.run_id,
                    "status": s.status,
                    "optimizer_model": s.optimizer_model,
                    "benchmark": s.benchmark,
                    "best_trial_id": s.best_trial_id,
                    "best_recipe_path": s.best_recipe_path,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                })
            })
            .collect();
        Ok(serde_json::to_string(&items).unwrap_or_else(|_| "[]".into()))
    }
}

// ---------------------------------------------------------------------------
// LLM Optimizer
// ---------------------------------------------------------------------------

#[pyclass(name = "LLMOptimizer")]
pub struct PyLLMOptimizer {
    inner: openjarvis_learning::optimize::LLMOptimizer,
}

#[pymethods]
impl PyLLMOptimizer {
    /// Create a new LLM optimizer.
    ///
    /// `search_space_json` is a JSON string representing the SearchSpace.
    /// `optimizer_model` is the model name to use for optimization proposals.
    #[new]
    #[pyo3(signature = (search_space_json, optimizer_model="gpt-4o"))]
    fn new(search_space_json: &str, optimizer_model: &str) -> PyResult<Self> {
        let search_space: openjarvis_learning::optimize::SearchSpace =
            serde_json::from_str(search_space_json)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let inner = openjarvis_learning::optimize::LLMOptimizer::new(
            search_space,
            optimizer_model.to_string(),
        );
        Ok(Self { inner })
    }

    /// Propose an initial configuration. Returns JSON string.
    fn propose_initial(&self) -> String {
        let config = self.inner.propose_initial();
        serde_json::to_string(&config).unwrap_or_else(|_| "{}".into())
    }

    /// Propose the next configuration based on trial history (JSON string).
    /// Returns JSON string.
    fn propose_next(&self, history_json: &str) -> PyResult<String> {
        let history: Vec<openjarvis_learning::optimize::TrialResult> =
            serde_json::from_str(history_json)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let config = self.inner.propose_next(&history);
        Ok(serde_json::to_string(&config).unwrap_or_else(|_| "{}".into()))
    }
}
