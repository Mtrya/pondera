use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[derive(Default)]
struct Node {
    edges: Vec<(u16, usize)>,
    token: Option<i32>,
}

/// The reversed trie lists entries ending at a position in increasing length.
#[pyclass(frozen, module = "pondera.chess._rules")]
pub(crate) struct MergeEncoder {
    roots: Vec<Option<usize>>,
    nodes: Vec<Node>,
}

#[pymethods]
impl MergeEncoder {
    #[new]
    fn new(entries: Vec<Vec<u16>>, base: u16) -> PyResult<Self> {
        let mut result = Self {
            roots: vec![None; usize::from(base)],
            nodes: Vec::new(),
        };
        for (index, atoms) in entries.into_iter().enumerate() {
            if !(2..=8).contains(&atoms.len()) || atoms.iter().any(|&atom| atom >= base) {
                return Err(PyValueError::new_err("Invalid merge expansion"));
            }
            let token = i32::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(i32::from(base)))
                .ok_or_else(|| PyValueError::new_err("Merge dictionary is too large"))?;
            let mut atoms = atoms.into_iter().rev();
            let first = usize::from(atoms.next().unwrap());
            let mut node = match result.roots[first] {
                Some(node) => node,
                None => {
                    let node = result.add_node();
                    result.roots[first] = Some(node);
                    node
                }
            };
            for atom in atoms {
                node = match result.nodes[node].edges.iter().find(|edge| edge.0 == atom) {
                    Some(edge) => edge.1,
                    None => {
                        let child = result.add_node();
                        result.nodes[node].edges.push((atom, child));
                        child
                    }
                };
            }
            if result.nodes[node].token.replace(token).is_some() {
                return Err(PyValueError::new_err("Duplicate merge expansion"));
            }
        }
        for node in &mut result.nodes {
            node.edges.sort_unstable_by_key(|edge| edge.0);
        }
        Ok(result)
    }

    fn encode<'py>(&self, py: Python<'py>, input: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let encoded = py
            .detach(|| self.encode_bytes(input))
            .ok_or_else(|| PyValueError::new_err("Invalid int32 atom stream"))?;
        Ok(PyBytes::new(py, &encoded))
    }
}

impl MergeEncoder {
    fn add_node(&mut self) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node::default());
        index
    }

    fn encode_bytes(&self, input: &[u8]) -> Option<Vec<u8>> {
        let chunks = input.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            return None;
        }
        let atoms: Vec<u16> = chunks
            .map(|chunk| {
                let atom = u16::try_from(i32::from_ne_bytes(chunk.try_into().unwrap())).ok()?;
                (usize::from(atom) < self.roots.len()).then_some(atom)
            })
            .collect::<Option<_>>()?;

        let mut distance = vec![0; atoms.len() + 1];
        let mut choices = Vec::with_capacity(atoms.len());
        for (index, &atom) in atoms.iter().enumerate() {
            let mut best = distance[index] + 1;
            let mut choice = (i32::from(atom), index);
            if let Some(mut node) = self.roots[usize::from(atom)] {
                for start in (index.saturating_sub(7)..index).rev() {
                    let edges = &self.nodes[node].edges;
                    let Ok(edge) = edges.binary_search_by_key(&atoms[start], |edge| edge.0) else {
                        break;
                    };
                    node = edges[edge].1;
                    if let Some(token) = self.nodes[node].token {
                        let candidate = distance[start] + 1;
                        // Longer matches arrive later, so a tie replaces the shorter choice.
                        if candidate <= best {
                            best = candidate;
                            choice = (token, start);
                        }
                    }
                }
            }
            distance[index + 1] = best;
            choices.push(choice);
        }
        let mut encoded = vec![0; distance[atoms.len()] * 4];
        let mut end = atoms.len();
        for chunk in encoded.chunks_exact_mut(4).rev() {
            let (token, start) = choices[end - 1];
            chunk.copy_from_slice(&token.to_ne_bytes());
            end = start;
        }
        Some(encoded)
    }
}
