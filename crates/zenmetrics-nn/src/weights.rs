//! Weight licence classes and the explicit opt-in for weights that are not
//! permissively licensed (`docs/PYIQA_PORT_PLAN.md`, "Weights").
//!
//! Each model type publishes its weights' terms as a `WEIGHTS` constant. A
//! model whose weights are [`WeightsLicence::Permissive`] loads with
//! `from_dir(dir, engine)`. Any other class takes a [`WeightsConsent`] that
//! names that constant, so using such weights is always a decision the
//! caller wrote down; the terms then bind the caller, not this crate.

use std::fmt;

use crate::Error;

/// How a model's trained weights are licensed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WeightsLicence {
    /// A permissive licence (BSD, MIT, Apache-2.0). Loads without consent.
    Permissive,
    /// The publisher states no licence for the weights. Loading needs a
    /// [`WeightsConsent`].
    Unknown,
    /// Noncommercial terms. Loading needs a [`WeightsConsent`].
    NonCommercial,
}

impl fmt::Display for WeightsLicence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Permissive => "permissive",
            Self::Unknown => "no stated licence",
            Self::NonCommercial => "noncommercial",
        })
    }
}

/// The published terms of one model's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightsTerms {
    /// The model's score column (`ckdn`, `topiq_fr`, …).
    pub model: &'static str,
    /// The licence class.
    pub licence: WeightsLicence,
    /// Where the terms (or the absence of any) can be read.
    pub terms_url: &'static str,
}

/// Weights terms the caller has accepted. Empty by default.
///
/// ```
/// # use zenmetrics_nn::{WeightsConsent, WeightsLicence, WeightsTerms};
/// const EXAMPLE: WeightsTerms = WeightsTerms {
///     model: "example",
///     licence: WeightsLicence::Unknown,
///     terms_url: "https://example.org/terms",
/// };
/// let consent = WeightsConsent::new().accept(EXAMPLE);
/// assert!(consent.accepts(&EXAMPLE));
/// ```
#[derive(Clone, Debug, Default)]
pub struct WeightsConsent {
    accepted: Vec<WeightsTerms>,
}

impl WeightsConsent {
    /// No terms accepted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accept `terms` (pass the model's `WEIGHTS` constant).
    #[must_use]
    pub fn accept(mut self, terms: WeightsTerms) -> Self {
        if !self.accepted.contains(&terms) {
            self.accepted.push(terms);
        }
        self
    }

    /// Whether `terms` may be loaded: always for permissive weights,
    /// otherwise only when these exact terms were accepted.
    pub fn accepts(&self, terms: &WeightsTerms) -> bool {
        terms.licence == WeightsLicence::Permissive || self.accepted.contains(terms)
    }

    /// `Ok` when [`Self::accepts`], else [`Error::WeightsNotAccepted`].
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "first caller is the CKDN port (nn-port lane)")
    )]
    pub(crate) fn check(&self, terms: &WeightsTerms) -> Result<(), Error> {
        if self.accepts(terms) {
            Ok(())
        } else {
            Err(Error::WeightsNotAccepted(*terms))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNKNOWN: WeightsTerms = WeightsTerms {
        model: "a",
        licence: WeightsLicence::Unknown,
        terms_url: "https://example.org/a",
    };
    const NC: WeightsTerms = WeightsTerms {
        model: "b",
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://example.org/b",
    };
    const FREE: WeightsTerms = WeightsTerms {
        model: "c",
        licence: WeightsLicence::Permissive,
        terms_url: "https://example.org/c",
    };

    #[test]
    fn permissive_needs_no_consent() {
        assert!(WeightsConsent::new().check(&FREE).is_ok());
    }

    #[test]
    fn other_classes_need_their_own_consent() {
        let none = WeightsConsent::new();
        assert!(matches!(
            none.check(&UNKNOWN),
            Err(Error::WeightsNotAccepted(t)) if t == UNKNOWN
        ));
        assert!(none.check(&NC).is_err());
        let one = WeightsConsent::new().accept(UNKNOWN);
        assert!(one.check(&UNKNOWN).is_ok());
        assert!(one.check(&NC).is_err(), "consent is per model");
        // Same model name, different terms: not accepted.
        let moved = WeightsTerms {
            terms_url: "https://example.org/changed",
            ..UNKNOWN
        };
        assert!(one.check(&moved).is_err());
    }

    #[test]
    fn refusal_message_names_the_terms() {
        let msg = WeightsConsent::new().check(&NC).unwrap_err().to_string();
        assert!(msg.contains("noncommercial") && msg.contains("https://example.org/b"));
    }
}
