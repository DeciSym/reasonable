//! Streaming RDFS inference against a fixed vocabulary.
//!
//! [`RdfsVocabulary`] takes the RDFS schema of a vocabulary (its `rdfs:subClassOf`,
//! `rdfs:subPropertyOf`, `rdfs:domain` and `rdfs:range` triples), and [`RdfsVocabulary::infer`]
//! derives what RDFS entails about one data triple from it: the triple for each super-property
//! of its predicate (rdfs7, rdfs5), the types its subject and object get from domains and ranges
//! (rdfs2, rdfs3), and the superclasses of those types and of any type the triple states (rdfs9,
//! rdfs11). Data triples are never treated as schema, so data can stream through, as
//! [`materialize_file`] does, in memory bounded by the vocabulary.
//!
//! This is the model of Apache Jena's `riot --rdfs=VOCAB`, made complete for these rules: riot
//! does not apply the domain or range of a super-property, does not add the superclasses of a type
//! inferred through a sub-property of `rdf:type`, and skips sub-property inference when the
//! vocabulary declares no subclasses. Like riot, ranges never type literals, so no inferred triple
//! has a literal subject.
//!
//! ```
//! use reasonable::rdfs::RdfsVocabulary;
//! use oxrdf::{NamedNode, Triple, vocab::{rdf, rdfs}};
//!
//! let iri = |s: &str| NamedNode::new_unchecked(s);
//! let vocabulary = RdfsVocabulary::new(vec![
//!     Triple::new(iri("urn:Dog"), rdfs::SUB_CLASS_OF, iri("urn:Animal")),
//! ]);
//! let fido = Triple::new(iri("urn:fido"), rdf::TYPE, iri("urn:Dog"));
//! let mut inferred = Vec::new();
//! vocabulary
//!     .infer(fido.as_ref(), |t| {
//!         inferred.push(t.into_owned());
//!         Ok::<_, std::convert::Infallible>(())
//!     })
//!     .unwrap();
//! assert_eq!(inferred, vec![Triple::new(iri("urn:fido"), rdf::TYPE, iri("urn:Animal"))]);
//! ```

use crate::common::{oxrdf_to_rio, rio_to_oxrdf, rio_to_oxrdf_ref};
use crate::error::{ReasonableError, Result};
use hashbrown::{HashMap, HashSet};
use oxrdf::vocab::{rdf, rdfs};
use oxrdf::{
    BlankNode, NamedNode, NamedOrBlankNode, NamedOrBlankNodeRef, Term, TermRef, Triple, TripleRef,
};
use rio_api::formatter::TriplesFormatter;
use rio_api::parser::TriplesParser;
use rio_turtle::{NTriplesFormatter, NTriplesParser, TurtleParser};
use std::fs;
use std::hash::Hash;
use std::io::{BufReader, Write};
use std::path::Path;

/// What RDFS entails about each triple of one property
#[derive(Default)]
struct PropertySchema {
    /// Its super-properties, transitively (rdfs5, rdfs7)
    super_properties: Vec<NamedNode>,
    /// Types of its subjects: the domains of the property and of its super-properties, with
    /// their superclasses (rdfs2, rdfs9)
    domain: Vec<NamedOrBlankNode>,
    /// Types of its objects that aren't literals, likewise from ranges (rdfs3, rdfs9)
    range: Vec<NamedOrBlankNode>,
    /// Whether it is `rdf:type` or a sub-property of it, so its object is a type of its subject
    is_type: bool,
}

/// The RDFS schema of a vocabulary, closed in advance so that what RDFS entails about a data
/// triple can be derived from that triple alone. See the [module documentation](self).
pub struct RdfsVocabulary {
    /// By property IRI
    properties: HashMap<String, PropertySchema>,
    /// The superclasses of each class IRI, transitively (rdfs9, rdfs11)
    superclasses: HashMap<String, Vec<NamedOrBlankNode>>,
}

/// What [`materialize_file`] read and wrote
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdfsCounts {
    /// Triples read from the input, all of which are written to the output
    pub input: usize,
    /// Triples inferred and written to the output. Different input triples can imply the same
    /// triple, in which case it is written more than once.
    pub inferred: usize,
}

impl RdfsVocabulary {
    /// Collects the schema of a vocabulary from its triples, ignoring all other triples
    pub fn new(triples: impl IntoIterator<Item = Triple>) -> Self {
        // Blank nodes are local to the vocabulary, so give them labels that can't be mistaken
        // for blank nodes in the data
        let mut bnodes: HashMap<BlankNode, BlankNode> = HashMap::new();
        let mut local = |node: NamedOrBlankNode| match node {
            NamedOrBlankNode::BlankNode(b) => NamedOrBlankNode::BlankNode(
                bnodes.entry(b).or_insert_with(BlankNode::default).clone(),
            ),
            node => node,
        };
        let mut subclass_of: HashMap<NamedOrBlankNode, Vec<NamedOrBlankNode>> = HashMap::new();
        let mut subproperty_of: HashMap<NamedNode, Vec<NamedNode>> = HashMap::new();
        let mut domains: HashMap<NamedNode, Vec<NamedOrBlankNode>> = HashMap::new();
        let mut ranges: HashMap<NamedNode, Vec<NamedOrBlankNode>> = HashMap::new();
        for Triple {
            subject,
            predicate,
            object,
        } in triples
        {
            let object = match object {
                Term::NamedNode(n) => NamedOrBlankNode::NamedNode(n),
                Term::BlankNode(b) => NamedOrBlankNode::BlankNode(b),
                _ => continue,
            };
            if predicate == rdfs::SUB_CLASS_OF {
                let class = local(subject);
                subclass_of.entry(class).or_default().push(local(object));
            } else if let NamedOrBlankNode::NamedNode(property) = subject {
                // the other statements describe properties, which have to be IRIs
                if predicate == rdfs::SUB_PROPERTY_OF {
                    if let NamedOrBlankNode::NamedNode(super_property) = object {
                        subproperty_of
                            .entry(property)
                            .or_default()
                            .push(super_property);
                    }
                } else if predicate == rdfs::DOMAIN {
                    domains.entry(property).or_default().push(local(object));
                } else if predicate == rdfs::RANGE {
                    ranges.entry(property).or_default().push(local(object));
                }
            }
        }

        // a list of classes, with all their superclasses
        let closed = |classes: &mut dyn Iterator<Item = &NamedOrBlankNode>| {
            let mut closed = Vec::new();
            for class in classes {
                closed.push(class.clone());
                closed.extend(reachable(&subclass_of, class));
            }
            sorted(closed)
        };
        let rdf_type = rdf::TYPE.into_owned();
        let described: HashSet<&NamedNode> = subproperty_of
            .keys()
            .chain(domains.keys())
            .chain(ranges.keys())
            .chain([&rdf_type])
            .collect();
        let mut properties = HashMap::new();
        for property in described {
            let mut super_properties = reachable(&subproperty_of, property);
            super_properties.retain(|p| p != property);
            let these = || std::iter::once(property).chain(&super_properties);
            let domain = closed(&mut these().flat_map(|p| domains.get(p)).flatten());
            let range = closed(&mut these().flat_map(|p| ranges.get(p)).flatten());
            let is_type = property == &rdf_type || super_properties.contains(&rdf_type);
            properties.insert(
                property.as_str().to_owned(),
                PropertySchema {
                    super_properties: sorted(super_properties),
                    domain,
                    range,
                    is_type,
                },
            );
        }
        let superclasses = subclass_of
            .keys()
            .filter_map(|class| match class {
                NamedOrBlankNode::NamedNode(iri) => {
                    let mut superclasses = reachable(&subclass_of, class);
                    superclasses.retain(|c| c != class);
                    Some((iri.as_str().to_owned(), sorted(superclasses)))
                }
                // only IRIs can be named in the data
                NamedOrBlankNode::BlankNode(_) => None,
            })
            .collect();
        RdfsVocabulary {
            properties,
            superclasses,
        }
    }

    /// Reads the vocabulary from a Turtle (`.ttl`) or N-Triples (`.nt`, `.n3`) file
    pub fn from_file(filename: impl AsRef<Path>) -> Result<Self> {
        let mut triples = Vec::new();
        parse_file(filename.as_ref(), |t| {
            triples.push(rio_to_oxrdf(t));
            Ok(())
        })?;
        Ok(Self::new(triples))
    }

    /// Calls `emit` with each triple that RDFS entails about `triple` given this vocabulary,
    /// other than `triple` itself, stopping at the first error `emit` returns. Each triple is
    /// emitted once per call, but separate calls can emit the same triple.
    pub fn infer<'a, E>(
        &'a self,
        triple: TripleRef<'a>,
        mut emit: impl FnMut(TripleRef<'a>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let Some(schema) = self.properties.get(triple.predicate.as_str()) else {
            return Ok(());
        };
        for property in &schema.super_properties {
            emit(TripleRef::new(triple.subject, property, triple.object))?;
        }
        let object = match triple.object {
            TermRef::NamedNode(n) => Some(NamedOrBlankNodeRef::NamedNode(n)),
            TermRef::BlankNode(b) => Some(NamedOrBlankNodeRef::BlankNode(b)),
            _ => None,
        };
        // (individual, class, whether to emit it) for each type found; the classes from the
        // schema come with their superclasses already
        let mut types = Vec::new();
        if let (true, Some(class)) = (schema.is_type, object) {
            // the triple states this type, or it was just emitted for rdf:type
            types.push((triple.subject, class, false));
            if let Some(superclasses) = self.superclasses.get(class_iri(class)) {
                types.extend(
                    superclasses
                        .iter()
                        .map(|c| (triple.subject, c.as_ref(), true)),
                );
            }
        }
        types.extend(
            schema
                .domain
                .iter()
                .map(|c| (triple.subject, c.as_ref(), true)),
        );
        if let Some(object) = object {
            types.extend(schema.range.iter().map(|c| (object, c.as_ref(), true)));
        }
        // what the vocabulary says about rdf:type itself also applies to the types found, though
        // vocabularies rarely say anything
        let rdf_type = &self.properties[rdf::TYPE.as_str()];
        let mut i = 0;
        while i < types.len() {
            let (individual, class, new) = types[i];
            i += 1;
            if types[..i - 1]
                .iter()
                .any(|&(x, c, _)| x == individual && c == class)
            {
                continue;
            }
            if new {
                emit(TripleRef::new(individual, rdf::TYPE, class))?;
                for property in &rdf_type.super_properties {
                    emit(TripleRef::new(individual, property, class))?;
                }
            }
            types.extend(
                rdf_type
                    .domain
                    .iter()
                    .map(|c| (individual, c.as_ref(), true)),
            );
            types.extend(rdf_type.range.iter().map(|c| (class, c.as_ref(), true)));
        }
        Ok(())
    }
}

/// Streams the triples of a Turtle (`.ttl`) or N-Triples (`.nt`, `.n3`) file to `output` as
/// N-Triples, each followed by what RDFS entails about it given `vocabulary`. Blank node labels
/// in the file are kept.
pub fn materialize_file(
    vocabulary: &RdfsVocabulary,
    filename: impl AsRef<Path>,
    output: impl Write,
) -> Result<RdfsCounts> {
    let mut formatter = NTriplesFormatter::new(output);
    let mut counts = RdfsCounts::default();
    parse_file(filename.as_ref(), |t| {
        formatter.format(&t)?;
        counts.input += 1;
        let (subject, predicate, object) = rio_to_oxrdf_ref(t);
        let subject = match subject {
            TermRef::NamedNode(n) => NamedOrBlankNodeRef::NamedNode(n),
            TermRef::BlankNode(b) => NamedOrBlankNodeRef::BlankNode(b),
            _ => unreachable!("rio subjects are IRIs or blank nodes"),
        };
        vocabulary.infer(TripleRef::new(subject, predicate, object), |inferred| {
            counts.inferred += 1;
            formatter.format(&oxrdf_to_rio(inferred))
        })?;
        Ok(())
    })?;
    formatter.finish()?.flush()?;
    Ok(counts)
}

/// Calls `on_triple` with each triple of a Turtle (`.ttl`) or N-Triples (`.nt`, `.n3`) file
fn parse_file(
    filename: &Path,
    mut on_triple: impl FnMut(rio_api::model::Triple) -> Result<()>,
) -> Result<()> {
    let f = BufReader::new(fs::File::open(filename)?);
    let mut on_triple = |t: rio_api::model::Triple| on_triple(t);
    match filename.extension().and_then(|e| e.to_str()) {
        Some("ttl") => TurtleParser::new(f, None).parse_all(&mut on_triple),
        Some("nt" | "n3") => NTriplesParser::new(f).parse_all(&mut on_triple),
        _ => Err(ReasonableError::IO(std::io::Error::other(
            "no parser for file (only ttl, nt and n3)",
        ))),
    }
}

/// The IRI of a class, or "" for a blank node, which no vocabulary class can match
fn class_iri<'a>(class: NamedOrBlankNodeRef<'a>) -> &'a str {
    match class {
        NamedOrBlankNodeRef::NamedNode(iri) => iri.as_str(),
        NamedOrBlankNodeRef::BlankNode(_) => "",
    }
}

/// The nodes reachable from `start` along `edges`, which include `start` on a cycle
fn reachable<N: Clone + Eq + Hash>(edges: &HashMap<N, Vec<N>>, start: &N) -> Vec<N> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<&N> = edges.get(start).into_iter().flatten().collect();
    while let Some(node) = stack.pop() {
        if seen.insert(node) {
            found.push(node.clone());
            stack.extend(edges.get(node).into_iter().flatten());
        }
    }
    found
}

/// `nodes` without duplicates, in a stable order so that output doesn't vary between runs
fn sorted<N: Eq + ToString>(mut nodes: Vec<N>) -> Vec<N> {
    nodes.sort_by_cached_key(|n| n.to_string());
    nodes.dedup();
    nodes
}
