// Benchmark-only extraction from the existing paired-NR result vector.
// No scorer, image conversion, public API or metric arithmetic lives here.
use std::cell::RefCell;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    distorted: f64,
    reference: f64,
}
fn extract(values: &[(&str, f64)]) -> Result<Pair, String> {
    if values.len() != 2 || values[0].0 != "brisque" || values[1].0 != "brisque_ref" {
        return Err("paired NR column/order mismatch".into());
    }
    let pair = Pair {
        distorted: values[0].1,
        reference: values[1].1,
    };
    if !pair.distorted.is_finite() || !pair.reference.is_finite() {
        return Err("nonfinite paired NR output".into());
    }
    Ok(pair)
}
#[derive(Default)]
struct Observed {
    first: Option<Pair>,
    last: Option<Pair>,
    count: u64,
    changed: bool,
}
impl Observed {
    fn record(&mut self, pair: Pair) -> f64 {
        let first = *self.first.get_or_insert(pair);
        self.changed |= first.distorted.to_bits() != pair.distorted.to_bits()
            || first.reference.to_bits() != pair.reference.to_bits();
        self.last = Some(pair);
        self.count += 1;
        pair.distorted
    }
    fn result(&self) -> Result<(f64, f64, u64), String> {
        if self.changed {
            return Err("paired NR result changed across calls".into());
        }
        let p = self.last.ok_or("missing paired NR result")?;
        Ok((p.distorted, p.reference, self.count))
    }
}
thread_local! { static OBSERVED: RefCell<Observed> = RefCell::new(Observed::default()); }
pub(crate) fn record(values: &[(&str, f64)]) -> Result<f64, String> {
    let pair = extract(values)?;
    Ok(OBSERVED.with(|o| o.borrow_mut().record(pair)))
}
pub(crate) fn result() -> Result<(f64, f64, u64), String> {
    OBSERVED.with(|o| o.borrow().result())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_both_and_preserves_headline() {
        let mut o = Observed::default();
        let p = extract(&[("brisque", 7.0), ("brisque_ref", 19.0)]).unwrap();
        assert_eq!(o.record(p), 7.0);
        assert_eq!(o.record(p), 7.0);
        assert_eq!(o.result().unwrap(), (7.0, 19.0, 2));
        assert_eq!(
            record(&[("brisque", 7.0), ("brisque_ref", 19.0)]).unwrap(),
            7.0
        );
        assert_eq!(result().unwrap(), (7.0, 19.0, 1));
    }
    #[test]
    fn refuses_missing_swapped_duplicate_and_nonfinite() {
        for values in [
            vec![],
            vec![("brisque", 1.0)],
            vec![("brisque_ref", 1.0), ("brisque", 2.0)],
            vec![("brisque", 1.0), ("brisque", 2.0)],
            vec![("brisque", 1.0), ("brisque_ref", f64::NAN)],
        ] {
            assert!(extract(&values).is_err());
        }
        assert!(Observed::default().result().is_err());
    }
    #[test]
    fn detects_reference_only_change() {
        let mut o = Observed::default();
        o.record(Pair {
            distorted: 1.0,
            reference: 2.0,
        });
        o.record(Pair {
            distorted: 1.0,
            reference: 3.0,
        });
        assert!(o.result().is_err());
    }
}
