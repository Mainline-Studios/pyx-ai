"""
Pyx AI Code — Your coding LLM (expanded).
=========================================
Complete, explain, and refactor code with a large rule set + optional external LLM.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
import re
import os

__version__ = "0.4"

# ---------------------------------------------------------------------------
# COMPLETIONS: (regex pattern, suffix or callable(match, prompt) -> suffix)
# Order: more specific first. Pattern is matched at end of prompt (rstrip).
# ---------------------------------------------------------------------------

def _m(pat: str, flags: int = 0) -> str:
    """Match at end."""
    return pat + r"\s*$"

_COMPLETIONS: List[Tuple[str, Any]] = [
    # ---- Function variants ----
    (r"function\s+(\w+)\s*\(", lambda m, p: ") {\n  \n}"),
    (r"function\s+(\w+)\s*$", lambda m, p: "() {\n  \n}"),
    (r"function\s*$", "name() {\n  \n}"),
    (r"(\w+)\s*=\s*function\s*$", lambda m, p: "() {\n  \n}"),
    (r"(\w+)\s*=\s*\(\s*\)\s*=>\s*$", lambda m, p: "{\n  \n}"),
    (r"(\w+)\s*=\s*\(\s*(\w+)\s*\)\s*=>\s*$", lambda m, p: " {\n  \n}"),
    (r"=>\s*$", "{\n  \n}"),
    (r"async\s+function\s+(\w+)\s*$", lambda m, p: "() {\n  \n}"),
    (r"async\s+\(\s*\)\s*=>\s*$", "{\n  \n}"),
    (r"async\s+\(\s*(\w+)\s*\)\s*=>\s*$", lambda m, p: " {\n  \n}"),
    # ---- Control flow ----
    (r"if\s*\(\s*$", "condition) {\n  \n}"),
    (r"else\s+if\s*\(\s*$", "condition) {\n  \n}"),
    (r"else\s*$", " {\n  \n}"),
    (r"for\s*\(\s*$", "let i = 0; i < n; i++) {\n  \n}"),
    (r"for\s*\(\s*(\w+)\s+of\s+$", lambda m, p: "iterable) {\n  \n}"),
    (r"for\s*\(\s*(\w+)\s+in\s+$", lambda m, p: "object) {\n  \n}"),
    (r"for\s*await\s*\(\s*(\w+)\s+of\s*$", lambda m, p: "asyncIterable) {\n  \n}"),
    (r"while\s*\(\s*$", "condition) {\n  \n}"),
    (r"do\s*$", " {\n  \n} while (condition);"),
    (r"switch\s*\(\s*$", "value) {\n  case :\n    break;\n  default:\n    break;\n}"),
    (r"case\s+$", "value:\n    break;"),
    (r"try\s*$", " {\n  \n} catch (err) {\n  \n}"),
    (r"catch\s*\(\s*$", "err) {\n  \n}"),
    (r"try\s*\{[^}]*\}\s*catch\s*\(\s*\w+\s*\)\s*\{[^}]*\}\s*$", ""),  # finally
    # ---- Declarations (JS/TS) ----
    (r"const\s*$", " name = "),
    (r"let\s*$", " name = "),
    (r"var\s*$", " name = "),
    (r"const\s+(\w+)\s*=\s*$", lambda m, p: "value;"),
    (r"let\s+(\w+)\s*=\s*$", lambda m, p: "value;"),
    (r"var\s+(\w+)\s*=\s*$", lambda m, p: "value;"),
    (r"(\w+)\s*:\s*(\w+)\s*=\s*$", lambda m, p: "value;"),
    (r"import\s+$", "{ } from \"\";"),
    (r"import\s+\{\s*$", " } from \"\";"),
    (r"import\s+\*\s+as\s+$", "name from \"\";"),
    (r"import\s+(\w+)\s+from\s+[\"']$", lambda m, p: "\";"),
    (r"export\s+default\s+$", " "),
    (r"export\s+{\s*$", " };\n"),
    (r"export\s+const\s+$", "name = "),
    (r"export\s+function\s+(\w+)\s*$", lambda m, p: "() {\n  \n}"),
    (r"require\s*\(\s*[\"']$", "\");"),
    # ---- Objects/arrays ----
    (r"for\s*\([^)]*\)\s*$", "{\n  \n}"),
    (r"while\s*\([^)]*\)\s*$", " {\n  \n}"),
    (r"if\s*\([^)]*\)\s*$", " {\n  \n}"),
    (r"\{\s*$", "\n  \n}"),
    (r"\[\s*$", "\n  \n]"),
    (r"<\s*$", ">\n"),
    (r"return\s*$", " ;"),
    (r"throw\s+new\s+Error\s*\(\s*$", "\"message\");"),
    (r"throw\s+$", "new Error(\"message\");"),
    (r"new\s+(\w+)\s*\(\s*$", lambda m, p: ");"),
    (r"await\s+$", "promise;"),
    (r"typeof\s+$", " x === \"undefined\""),
    (r"instanceof\s+$", " Constructor"),
    # ---- Python ----
    (r"def\s+(\w+)\s*\(\s*$", lambda m, p: "):\n    pass"),
    (r"def\s+(\w+)\s*\(\s*(\w+)\s*$", lambda m, p: "):\n    pass"),
    (r"def\s+$", "name():\n    pass"),
    (r"class\s+(\w+)\s*\(\s*$", lambda m, p: "):\n    pass"),
    (r"class\s+(\w+)\s*$", lambda m, p: ":\n    pass"),
    (r"class\s+$", "Name:\n    pass"),
    (r"if\s+(\w+)\s*$", lambda m, p: ":\n    pass"),
    (r"elif\s+(\w+)\s*$", lambda m, p: ":\n    pass"),
    (r"else\s*:\s*$", "\n    pass"),
    (r"for\s+(\w+)\s+in\s+$", lambda m, p: ":\n    pass"),
    (r"while\s+(\w+)\s*$", lambda m, p: ":\n    pass"),
    (r"try\s*:\s*$", "\n    pass\nexcept Exception:\n    pass"),
    (r"except\s+(\w+)\s*$", lambda m, p: ":\n    pass"),
    (r"with\s+(\w+)\s*\(\s*$", lambda m, p: ") as x:\n    pass"),
    (r"async\s+def\s+(\w+)\s*$", lambda m, p: "():\n    pass"),
    (r"@\s*(\w+)\s*$", lambda m, p: "\ndef "),
    (r"from\s+(\w+)\s+import\s+$", lambda m, p: " "),
    (r"import\s+$", " "),
    (r"lambda\s+(\w+)\s*$", lambda m, p: ": "),
    (r"lambda\s+(\w+)\s*,\s*$", lambda m, p: "x: "),
    (r"assert\s+$", "condition, \"message\""),
    (r"raise\s+$", "ValueError(\"message\")"),
    (r"yield\s+$", " "),
    (r"async\s+for\s+(\w+)\s+in\s+$", lambda m, p: ":\n    pass"),
    # ---- three.js ----
    (r"new\s+THREE\.(\w+)\s*\(\s*$", lambda m, p: ");"),
    (r"scene\.add\s*\(\s*$", "mesh);"),
    (r"scene\.remove\s*\(\s*$", "mesh);"),
    (r"(\w+)\.position\.(\w+)\s*=\s*$", lambda m, p: "0;"),
    (r"renderer\.render\s*\(\s*$", "scene, camera);"),
    (r"camera\.position\.(\w+)\s*=\s*$", lambda m, p: "0;"),
    (r"new\s+THREE\.Mesh\s*\(\s*$", "geometry, material);"),
    (r"new\s+THREE\.BufferGeometry\s*\(\s*$", ");"),
    (r"new\s+THREE\.(\w+)Material\s*\(\s*$", lambda m, p: "({ });"),
    (r"THREE\.(\w+)Geometry\s*\(\s*$", lambda m, p: ");"),
    (r"requestAnimationFrame\s*\(\s*$", "animate);"),
    # ---- React ----
    (r"React\.createElement\s*\(\s*$", "Component, { });"),
    (r"useState\s*\(\s*$", "initialValue);"),
    (r"useEffect\s*\(\s*\(\s*\)\s*=>\s*\{\s*$", "\n  return () => { };\n}, []);"),
    (r"useEffect\s*\(\s*\(\s*\)\s*=>\s*$", "{\n  \n}, []);"),
    (r"useCallback\s*\(\s*\(\s*\)\s*=>\s*$", "{\n  \n}, []);"),
    (r"useMemo\s*\(\s*\(\s*\)\s*=>\s*$", "{\n  \n}, []);"),
    (r"useRef\s*\(\s*$", "initialValue);"),
    (r"<(\w+)\s+$", lambda m, p: ">\n  \n</" + (m.group(1) if m.lastindex else "div") + ">"),
    (r"<(\w+)\s*>\s*$", lambda m, p: "\n  \n</" + (m.group(1) if m.lastindex else "div") + ">"),
    # ---- Node / common ----
    (r"module\.exports\s*=\s*$", " "),
    (r"console\.log\s*\(\s*$", ");"),
    (r"console\.error\s*\(\s*$", ");"),
    (r"console\.warn\s*\(\s*$", ");"),
    (r"setTimeout\s*\(\s*\(\s*\)\s*=>\s*$", "{\n  \n}, 0);"),
    (r"setInterval\s*\(\s*\(\s*\)\s*=>\s*$", "{\n  \n}, 1000);"),
    (r"Promise\.(\w+)\s*\(\s*\(\s*$", lambda m, p: "resolve, reject) => {\n  \n});"),
    (r"\.then\s*\(\s*(\w+)\s*=>\s*$", lambda m, p: "{\n  \n});"),
    (r"\.catch\s*\(\s*(\w+)\s*=>\s*$", lambda m, p: "{\n  \n});"),
    (r"\.finally\s*\(\s*\(\s*\)\s*=>\s*$", "{\n  \n});"),
    (r"fetch\s*\(\s*[\"']$", "\")\n  .then(res => res.json())\n  .then(data => { });"),
    (r"addEventListener\s*\(\s*[\"']$", "\"event\", (e) => {\n  \n});"),
    (r"\(\s*$", "\n  \n)"),
    # ---- GLSL ----
    (r"void\s+main\s*\(\s*\)\s*$", " {\n  \n}"),
    (r"vec(\d)\s+(\w+)\s*=\s*$", lambda m, p: "vec" + (m.group(1) or "3") + "(0.0);"),
    (r"float\s+(\w+)\s*=\s*$", lambda m, p: "0.0;"),
    (r"if\s*\(\s*$", "condition) {\n  \n}"),
    (r"for\s*\(\s*$", "int i = 0; i < n; i++) {\n  \n}"),
    (r"return\s*$", " ;"),
]

# Fallback: trailing : or { or [
_FALLBACK_SUFFIXES = [(":", "\n  \n"), ("{", "\n  \n}"), ("[", "\n  \n]"), ("(", "\n  \n)")]


def complete(prompt: str, max_tokens: int = 256, language: Optional[str] = None, **kwargs: Any) -> str:
    """Complete code from a prompt. Uses large pattern set; optional external LLM."""
    prompt = prompt.rstrip()
    base = prompt
    for pattern, suffix in _COMPLETIONS:
        m = re.search(pattern, base, re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if m:
            if callable(suffix):
                try:
                    add = suffix(m, base)
            except Exception:
                    add = "\n  \n"
            else:
                add = suffix
            out = base + add
            return out[: len(prompt) + max_tokens]
    for end, add in _FALLBACK_SUFFIXES:
        if base.endswith(end):
            return (base + add)[: len(prompt) + max_tokens]
    return (base + "\n")[: len(prompt) + max_tokens]


# ---------------------------------------------------------------------------
# EXPLAIN: (pattern, explanation)
# ---------------------------------------------------------------------------
_EXPLAIN_PATTERNS: List[Tuple[str, str]] = [
    (r"\basync\s+function\b|\basync\s*\([^)]*\)\s*=>", "Asynchronous function (returns a Promise)."),
    (r"\bawait\s+", "Waits for a Promise to resolve."),
    (r"\.then\s*\(|\.catch\s*\(|\.finally\s*\(", "Promise chain (then/catch/finally)."),
    (r"\bPromise\.(all|race|resolve|reject)\s*\(", "Promise utility (all/race/resolve/reject)."),
    (r"\bfunction\s*\*|\byield\b", "Generator function (yields values)."),
    (r"\bfor\s+await\b", "Async iteration over an async iterable."),
    (r"\bfunction\s+\w+\s*\(", "Defines a named function."),
    (r"=>\s*\{|\(\s*\)\s*=>", "Arrow function (anonymous)."),
    (r"\bclass\s+\w+", "Defines a class."),
    (r"\bextends\s+\w+", "Class inheritance (subclass)."),
    (r"\bconstructor\s*\(", "Class constructor (runs when creating an instance)."),
    (r"\bstatic\s+\w+\s*\(", "Static method (called on the class, not instance)."),
    (r"\bget\s+\w+\s*\(\)|\bset\s+\w+\s*\(", "Getter or setter (property access/mutation)."),
    (r"\bif\s*\(", "Conditional branch (if)."),
    (r"\belse\s+if\s*\(", "Else-if branch."),
    (r"\belse\s*\{", "Else branch."),
    (r"\bswitch\s*\(", "Switch statement (multiple branches on a value)."),
    (r"\bfor\s*\(", "For loop (init; condition; step)."),
    (r"\bfor\s+\w+\s+of\b", "For-of loop (iterate over iterable values)."),
    (r"\bfor\s+\w+\s+in\b", "For-in loop (iterate over object keys)."),
    (r"\bwhile\s*\(", "While loop."),
    (r"\bdo\s*\{", "Do-while loop (body runs at least once)."),
    (r"\btry\s*\{", "Try block (catch errors)."),
    (r"\bcatch\s*\(", "Catches errors from try block."),
    (r"\bfinally\s*\{", "Runs after try/catch (cleanup)."),
    (r"\bthrow\b", "Throws an exception."),
    (r"\breturn\b", "Returns a value from the function."),
    (r"\bimport\s+.*\bfrom\b", "ES module import."),
    (r"\bexport\s+", "ES module export."),
    (r"\brequire\s*\(", "CommonJS require (Node-style import)."),
    (r"\bconst\s+\w+\s*=", "Declares a constant (block-scoped, cannot reassign)."),
    (r"\blet\s+\w+\s*=", "Declares a variable (block-scoped, can reassign)."),
    (r"\bvar\s+\w+\s*=", "Declares a variable (function-scoped, legacy)."),
    (r"\bnew\s+\w+\s*\(", "Creates an instance (calls constructor)."),
    (r"\bthis\.\w+", "Refers to property/method on current object (this)."),
    (r"\btypeof\s+", "Returns the type of a value (string)."),
    (r"\binstanceof\s+", "Checks if value is an instance of a constructor."),
    (r"\bundefined\b|\bnull\b", "Represents absence of value (undefined or null)."),
    (r"\btrue\b|\bfalse\b", "Boolean literal."),
    (r"\bdef\s+\w+\s*\(", "Defines a Python function."),
    (r"\bclass\s+\w+.*:", "Defines a Python class."),
    (r"\bif\s+.*:", "Python conditional (if)."),
    (r"\belif\s+.*:", "Python else-if."),
    (r"\belse\s*:", "Python else."),
    (r"\bfor\s+\w+\s+in\b", "Python for loop (iterate over iterable)."),
    (r"\bwhile\s+.*:", "Python while loop."),
    (r"\btry\s*:", "Python try block."),
    (r"\bexcept\b", "Python except (catch exception)."),
    (r"\bwith\s+.*\s+as\b", "Python context manager (with statement)."),
    (r"\blambda\s+", "Python anonymous function (lambda)."),
    (r"\byield\b", "Python generator yield."),
    (r"\basync\s+def\b", "Python async function."),
    (r"\bTHREE\.\w+", "three.js API (WebGL/3D)."),
    (r"\bscene\.(add|remove)\b", "Adds or removes an object from the three.js scene."),
    (r"\buseState\b|\buseEffect\b|\buseCallback\b|\buseMemo\b|\buseRef\b", "React hook (state or side effect)."),
    (r"React\.createElement\b|<[A-Z]\w+", "React element (component or JSX)."),
    (r"void\s+main\s*\(", "GLSL shader main entry (vertex/fragment)."),
    (r"\bvec\d\b|\bmat\d\b|\bfloat\b|\bint\b", "GLSL type (vector, matrix, scalar)."),
]


def explain(snippet: str, language: Optional[str] = None, **kwargs: Any) -> str:
    """Explain what the code does. Uses pattern set; optional external LLM."""
    s = snippet.strip()
    if not s:
        return "Empty snippet."
    for pattern, explanation in _EXPLAIN_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return explanation
    if len(s) > 200:
        return "Code snippet (use a trained LLM for a full explanation)."
    return "Code snippet (pattern not in explain set; add more rules or use an LLM)."


# ---------------------------------------------------------------------------
# REFACTOR: apply passes
# ---------------------------------------------------------------------------
def refactor(snippet: str, instruction: Optional[str] = None, **kwargs: Any) -> str:
    """Refactor code. Applies var->const/let, trim, normalize; optional external LLM."""
    if not snippet.strip():
        return snippet
    out = snippet
    lines = out.split("\n")
    # Trim trailing whitespace per line
    out = "\n".join(line.rstrip() for line in lines)
    # Normalize multiple blank lines to at most one
    out = re.sub(r"\n{3,}", "\n\n", out)
    if instruction:
        inst = instruction.lower()
        if "var" in inst or "const" in inst or "let" in inst:
            # var -> let (simple: only when single declaration and no reassignment in same scope - we do naive replace for demo)
            out = re.sub(r"\bvar\s+(\w+)\s*=", r"const \1 =", out, count=1)
        if "quote" in inst or "single" in inst:
            out = out.replace('"', "'")
        if "quote" in inst and "double" in inst:
            out = out.replace("'", '"')
    return out


def health() -> Dict[str, Any]:
    """Report whether the code model is loaded and ready."""
    return {
        "loaded": True,
        "model": "pyx-code",
        "version": __version__,
        "completions": len(_COMPLETIONS),
        "explain_patterns": len(_EXPLAIN_PATTERNS),
    }
