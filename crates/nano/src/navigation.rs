//! Rust syntax navigation over source returned by the caller; no filesystem access.
use proc_macro2::{Ident, Span};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use syn::visit::{self, Visit};

#[derive(Clone, Debug, PartialEq)]
pub enum NavigationKind {
    Definitions,
    References,
    Callers,
    Implementations,
}
impl NavigationKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Definitions => "symbol_definitions",
            Self::References => "symbol_references",
            Self::Callers => "symbol_callers",
            Self::Implementations => "symbol_implementations",
        }
    }
}

pub fn valid_symbol(symbol: &str) -> bool {
    let symbol = symbol.strip_prefix("r#").unwrap_or(symbol);
    !symbol.is_empty()
        && symbol != "_"
        && symbol
            .bytes()
            .enumerate()
            .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || i > 0 && c.is_ascii_digit())
}

pub fn parse_request(prompt: &str) -> Option<Result<(NavigationKind, String), String>> {
    let trimmed = prompt.trim();
    let lower = trimmed.to_ascii_lowercase();
    let prefixes = [
        ("--definitions", NavigationKind::Definitions),
        ("--references", NavigationKind::References),
        ("--callers", NavigationKind::Callers),
        ("--implementations", NavigationKind::Implementations),
        ("who calls ", NavigationKind::Callers),
        ("callers of ", NavigationKind::Callers),
        ("find call sites of ", NavigationKind::Callers),
        ("find references to ", NavigationKind::References),
        ("find implementations of ", NavigationKind::Implementations),
        ("implementations of ", NavigationKind::Implementations),
        ("who implements ", NavigationKind::Implementations),
    ];
    for (prefix, kind) in prefixes {
        if let Some(rest) = lower.strip_prefix(prefix) {
            if prefix.starts_with("--") && !rest.is_empty() && !rest.starts_with([' ', '\t', ':']) {
                continue;
            }
            let symbol = trimmed[prefix.len()..]
                .trim_start_matches(':')
                .trim()
                .trim_matches(['`', '"', '\'', '?']);
            return Some(if valid_symbol(symbol) {
                Ok((kind, symbol.to_string()))
            } else {
                Err("Specify one Rust identifier, for example --callers parse_atomic. Qualified paths and expressions are not supported.".into())
            });
        }
    }
    None
}

fn name(ident: &Ident) -> String {
    ident.to_string().trim_start_matches("r#").to_string()
}
fn path_name(path: &syn::Path) -> Option<String> {
    path.segments.last().map(|s| name(&s.ident))
}
fn type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) => path_name(&p.path),
        _ => None,
    }
}

struct Index<'a> {
    source: &'a str,
    file: &'a str,
    symbol: &'a str,
    definitions: Vec<Value>,
    references: Vec<Value>,
    callers: Vec<Value>,
    implementations: Vec<Value>,
    declarations: BTreeSet<(usize, usize)>,
    owner: Option<String>,
    impl_owner: Option<String>,
}
impl Index<'_> {
    fn location(&self, span: Span, kind: &str) -> Value {
        let start = span.start();
        json!({"path":self.file,"line":start.line,"column":start.column + 1,"kind":kind,
            "excerpt":self.source.lines().nth(start.line.saturating_sub(1)).unwrap_or("")})
    }
    fn define(&mut self, ident: &Ident, kind: &str) {
        if name(ident) == self.symbol {
            let start = ident.span().start();
            self.declarations.insert((start.line, start.column + 1));
            self.definitions.push(self.location(ident.span(), kind));
        }
    }
    fn call(&mut self, ident: &Ident, kind: &str) {
        if name(ident) == self.symbol {
            let mut call = self.location(ident.span(), kind);
            call["caller"] = json!(self.owner);
            self.callers.push(call);
        }
    }
    fn function_owner(&self, ident: &Ident) -> String {
        match &self.impl_owner {
            Some(owner) => format!("{owner}::{}", name(ident)),
            None => name(ident),
        }
    }
}
impl<'ast> Visit<'ast> for Index<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        match item {
            syn::Item::Struct(i) => self.define(&i.ident, "struct"),
            syn::Item::Enum(i) => self.define(&i.ident, "enum"),
            syn::Item::Trait(i) => self.define(&i.ident, "trait"),
            syn::Item::Type(i) => self.define(&i.ident, "type"),
            syn::Item::Const(i) => self.define(&i.ident, "const"),
            syn::Item::Static(i) => self.define(&i.ident, "static"),
            syn::Item::Mod(i) => self.define(&i.ident, "module"),
            syn::Item::Macro(i) => {
                if let Some(ident) = &i.ident {
                    self.define(ident, "macro");
                }
            }
            _ => {}
        }
        visit::visit_item(self, item);
    }
    fn visit_signature(&mut self, sig: &'ast syn::Signature) {
        self.define(&sig.ident, "function");
        visit::visit_signature(self, sig);
    }
    fn visit_variant(&mut self, item: &'ast syn::Variant) {
        self.define(&item.ident, "enum_variant");
        visit::visit_variant(self, item);
    }
    fn visit_field(&mut self, item: &'ast syn::Field) {
        if let Some(ident) = &item.ident {
            self.define(ident, "field");
        }
        visit::visit_field(self, item);
    }
    fn visit_impl_item_const(&mut self, item: &'ast syn::ImplItemConst) {
        self.define(&item.ident, "associated_const");
        visit::visit_impl_item_const(self, item);
    }
    fn visit_impl_item_type(&mut self, item: &'ast syn::ImplItemType) {
        self.define(&item.ident, "associated_type");
        visit::visit_impl_item_type(self, item);
    }
    fn visit_trait_item_const(&mut self, item: &'ast syn::TraitItemConst) {
        self.define(&item.ident, "associated_const");
        visit::visit_trait_item_const(self, item);
    }
    fn visit_trait_item_type(&mut self, item: &'ast syn::TraitItemType) {
        self.define(&item.ident, "associated_type");
        visit::visit_trait_item_type(self, item);
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        let previous = self.owner.replace(self.function_owner(&item.sig.ident));
        visit::visit_item_fn(self, item);
        self.owner = previous;
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let previous = self.owner.replace(self.function_owner(&item.sig.ident));
        visit::visit_impl_item_fn(self, item);
        self.owner = previous;
    }
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        let previous = self.owner.replace(self.function_owner(&item.sig.ident));
        visit::visit_trait_item_fn(self, item);
        self.owner = previous;
    }
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let receiver = type_name(&item.self_ty);
        let trait_name = item
            .trait_
            .as_ref()
            .and_then(|(_, path, _)| path_name(path));
        if receiver.as_deref() == Some(self.symbol) || trait_name.as_deref() == Some(self.symbol) {
            let mut location = self.location(item.impl_token.span, "impl");
            location["self_type"] = json!(receiver);
            location["trait"] = json!(trait_name);
            location["negative"] = json!(
                item.trait_
                    .as_ref()
                    .is_some_and(|(neg, _, _)| neg.is_some())
            );
            self.implementations.push(location);
        }
        let previous = std::mem::replace(&mut self.impl_owner, receiver);
        visit::visit_item_impl(self, item);
        self.impl_owner = previous;
    }
    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        let previous = self.impl_owner.replace(name(&item.ident));
        visit::visit_item_trait(self, item);
        self.impl_owner = previous;
    }
    fn visit_ident(&mut self, ident: &'ast Ident) {
        if name(ident) == self.symbol {
            self.references
                .push(self.location(ident.span(), "identifier_usage"));
        }
    }
    fn visit_lifetime(&mut self, _lifetime: &'ast syn::Lifetime) {}
    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*expr.func {
            if let Some(segment) = path.path.segments.last() {
                self.call(&segment.ident, "function_call");
            }
        }
        visit::visit_expr_call(self, expr);
    }
    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        self.call(&expr.method, "method_call");
        visit::visit_expr_method_call(self, expr);
    }
}

pub fn analyze(
    file: &str,
    source: &str,
    symbol: &str,
    kind: &NavigationKind,
) -> Result<Vec<Value>, String> {
    let ast =
        syn::parse_file(source).map_err(|e| format!("Rust syntax could not be parsed: {e}"))?;
    let mut index = Index {
        source,
        file,
        symbol: symbol.trim_start_matches("r#"),
        definitions: vec![],
        references: vec![],
        callers: vec![],
        implementations: vec![],
        declarations: BTreeSet::new(),
        owner: None,
        impl_owner: None,
    };
    index.visit_file(&ast);
    Ok(match kind {
        NavigationKind::Definitions => index.definitions,
        NavigationKind::References => index
            .references
            .into_iter()
            .filter(|r| {
                !index.declarations.contains(&(
                    r["line"].as_u64().unwrap() as usize,
                    r["column"].as_u64().unwrap() as usize,
                ))
            })
            .collect(),
        NavigationKind::Callers => index.callers,
        NavigationKind::Implementations => index.implementations,
    })
}

/// Read windows must identify their position and total size. Unlabelled text
/// is not treated as proof that a full file was returned.
pub fn read_window(text: &str, expected_start: usize) -> Result<(&str, usize, usize), String> {
    let (header, source) = text
        .split_once('\n')
        .ok_or("read_file result has no line-range header")?;
    let header = header
        .strip_prefix("Lines ")
        .ok_or("read_file result has no line-range header")?;
    let (range, total) = header
        .split_once(" of ")
        .ok_or("invalid read_file line-range header")?;
    let (start, end) = range
        .split_once('-')
        .ok_or("invalid read_file line range")?;
    let start: usize = start.parse().map_err(|_| "invalid start line")?;
    let end: usize = end.parse().map_err(|_| "invalid end line")?;
    let total: usize = total
        .trim_end_matches(':')
        .parse()
        .map_err(|_| "invalid total lines")?;
    if start != expected_start
        || end < start
        || end > total
        || source.lines().count() != end - start + 1
    {
        return Err("read_file returned inconsistent or truncated line coverage".into());
    }
    Ok((source, end, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = r#"
// fn target() {} and target() in comments are not evidence.
pub trait Service { fn target(&self); }
pub struct Worker;
impl Service for Worker {
    fn target(&self) {}
    fn work(&self) { self.target(); helper::target(); }
}
impl Worker { fn local(&self) { self.target(); } }
fn target() {}
fn target_extra() {}
fn outer() {
    let _message = "target()";
    target();
    target_extra();
    let pointer = target;
    let _ = pointer;
}
"#;
    #[test]
    fn definitions_and_references_use_identifiers_not_substrings_or_comments() {
        let defs = analyze("src/lib.rs", SOURCE, "target", &NavigationKind::Definitions).unwrap();
        assert_eq!(defs.len(), 3); // Trait declaration, impl method, free function.
        let refs = analyze("src/lib.rs", SOURCE, "target", &NavigationKind::References).unwrap();
        assert_eq!(refs.len(), 5); // Four calls and a function pointer.
        assert!(
            refs.iter()
                .all(|r| r["excerpt"].as_str().unwrap() != "fn target_extra() {}")
        );
        assert!(refs.iter().all(|r| {
            !defs
                .iter()
                .any(|d| d["line"] == r["line"] && d["column"] == r["column"])
        }));
    }
    #[test]
    fn callers_distinguish_calls_from_other_references_and_include_owners() {
        let calls = analyze("src/lib.rs", SOURCE, "target", &NavigationKind::Callers).unwrap();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0]["caller"], "Worker::work");
        assert_eq!(calls[0]["kind"], "method_call");
        assert_eq!(calls[1]["kind"], "function_call");
        assert_eq!(calls[3]["caller"], "outer");
    }
    #[test]
    fn implementations_find_trait_and_inherent_blocks() {
        let trait_impls = analyze(
            "src/lib.rs",
            SOURCE,
            "Service",
            &NavigationKind::Implementations,
        )
        .unwrap();
        assert_eq!(trait_impls.len(), 1);
        assert_eq!(trait_impls[0]["self_type"], "Worker");
        assert_eq!(trait_impls[0]["trait"], "Service");
        let type_impls = analyze(
            "src/lib.rs",
            SOURCE,
            "Worker",
            &NavigationKind::Implementations,
        )
        .unwrap();
        assert_eq!(type_impls.len(), 2);
        assert_eq!(type_impls[1]["trait"], Value::Null);
        let generic = "impl<T> Service for Wrapper<T> { fn target(&self) {} }";
        assert_eq!(
            analyze(
                "lib.rs",
                generic,
                "Wrapper",
                &NavigationKind::Implementations
            )
            .unwrap()
            .len(),
            1
        );
    }
    #[test]
    fn raw_identifiers_multiline_syntax_and_real_locations_are_supported() {
        let source = "pub(crate) async fn\nr#type() {}\nfn caller() { r#type(); }\n";
        let defs = analyze("lib.rs", source, "r#type", &NavigationKind::Definitions).unwrap();
        assert_eq!(defs[0]["line"], 2);
        assert_eq!(defs[0]["column"], 1);
        let calls = analyze("lib.rs", source, "r#type", &NavigationKind::Callers).unwrap();
        assert_eq!(calls[0]["line"], 3);
        assert_eq!(calls[0]["caller"], "caller");
        assert!(
            analyze(
                "lib.rs",
                "fn unfinished(",
                "unfinished",
                &NavigationKind::Definitions
            )
            .is_err()
        );
    }
    #[test]
    fn windows_require_complete_consistent_ranges() {
        assert_eq!(
            read_window("Lines 301-302 of 302:\na\nb\n", 301).unwrap(),
            ("a\nb\n", 302, 302)
        );
        for text in [
            "fn example() {}",
            "Lines 1-3 of 3:\na\nb",
            "Lines 2-3 of 3:\na\nb",
            "Lines 1-5 of 3:\na",
        ] {
            assert!(read_window(text, 1).is_err());
        }
    }
    #[test]
    fn declarations_cover_fields_variants_and_associated_items() {
        let source = "struct S { value: u8 } enum E { Choice } trait T { type Output; const LIMIT: u8; } impl T for S { type Output = u8; const LIMIT: u8 = 1; }";
        for (symbol, count) in [("value", 1), ("Choice", 1), ("Output", 2), ("LIMIT", 2)] {
            assert_eq!(
                analyze("lib.rs", source, symbol, &NavigationKind::Definitions)
                    .unwrap()
                    .len(),
                count
            );
            assert!(
                analyze("lib.rs", source, symbol, &NavigationKind::References)
                    .unwrap()
                    .is_empty()
            );
        }
    }
    #[test]
    fn directives_reject_expressions_and_qualified_paths() {
        for prompt in [
            "--callers",
            "--references foo::bar",
            "--implementations Foo; command",
            "--definitions ../foo",
        ] {
            assert!(parse_request(prompt).unwrap().is_err());
        }
        assert_eq!(
            parse_request("who calls target?").unwrap().unwrap(),
            (NavigationKind::Callers, "target".into())
        );
        assert!(parse_request("--callers-extra").is_none());
    }
}
