/**
 * Pyx Assistant — local math + unit conversion (no LLM).
 */
(function (root) {
  "use strict";

  var WORDS = {
    zero: 0, oh: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7,
    eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14,
    fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20,
    thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90,
    hundred: 100, thousand: 1000, million: 1000000, billion: 1000000000,
    a: 1, an: 1, half: 0.5, quarter: 0.25, thirds: 1 / 3, third: 1 / 3,
    pi: Math.PI, e: Math.E,
  };

  var OPS = { plus: "+", minus: "-", times: "*", multiplied: "*", over: "/", divided: "/", into: "*" };

  var UNITS = {
    km: { dim: "length", to: 1000 },
    kilometer: { dim: "length", to: 1000 },
    kilometers: { dim: "length", to: 1000 },
    m: { dim: "length", to: 1 },
    meter: { dim: "length", to: 1 },
    meters: { dim: "length", to: 1 },
    cm: { dim: "length", to: 0.01 },
    millimeter: { dim: "length", to: 0.001 },
    mm: { dim: "length", to: 0.001 },
    mile: { dim: "length", to: 1609.344 },
    miles: { dim: "length", to: 1609.344 },
    mi: { dim: "length", to: 1609.344 },
    yard: { dim: "length", to: 0.9144 },
    yards: { dim: "length", to: 0.9144 },
    foot: { dim: "length", to: 0.3048 },
    feet: { dim: "length", to: 0.3048 },
    ft: { dim: "length", to: 0.3048 },
    inch: { dim: "length", to: 0.0254 },
    inches: { dim: "length", to: 0.0254 },
    kg: { dim: "mass", to: 1 },
    kilogram: { dim: "mass", to: 1 },
    kilograms: { dim: "mass", to: 1 },
    g: { dim: "mass", to: 0.001 },
    gram: { dim: "mass", to: 0.001 },
    grams: { dim: "mass", to: 0.001 },
    lb: { dim: "mass", to: 0.45359237 },
    lbs: { dim: "mass", to: 0.45359237 },
    pound: { dim: "mass", to: 0.45359237 },
    pounds: { dim: "mass", to: 0.45359237 },
    oz: { dim: "mass", to: 0.028349523125 },
    ounce: { dim: "mass", to: 0.028349523125 },
    ounces: { dim: "mass", to: 0.028349523125 },
    c: { dim: "temp", to: "c" },
    f: { dim: "temp", to: "f" },
    k: { dim: "temp", to: "k" },
    celsius: { dim: "temp", to: "c" },
    fahrenheit: { dim: "temp", to: "f" },
    kelvin: { dim: "temp", to: "k" },
    l: { dim: "vol", to: 1 },
    liter: { dim: "vol", to: 1 },
    liters: { dim: "vol", to: 1 },
    ml: { dim: "vol", to: 0.001 },
    gallon: { dim: "vol", to: 3.785411784 },
    gallons: { dim: "vol", to: 3.785411784 },
    cup: { dim: "vol", to: 0.2365882365 },
    cups: { dim: "vol", to: 0.2365882365 },
  };

  function nice(n) {
    if (typeof n !== "number" || !isFinite(n)) return null;
    var r = Math.round(n * 1e10) / 1e10;
    if (Math.abs(r - Math.round(r)) < 1e-10) return String(Math.round(r));
    return String(parseFloat(r.toPrecision(10)));
  }

  function wordsToNumber(tokens) {
    var total = 0;
    var current = 0;
    var used = false;
    tokens.forEach(function (tok) {
      var w = tok.toLowerCase();
      if (w === "and") return;
      if (WORDS[w] == null) return;
      used = true;
      var v = WORDS[w];
      if (v === 100 || v === 1000 || v === 1000000 || v === 1000000000) {
        if (current === 0) current = 1;
        current *= v;
        if (v >= 1000) {
          total += current;
          current = 0;
        }
      } else {
        current += v;
      }
    });
    if (!used) return null;
    return total + current;
  }

  function rewriteWords(text) {
    var raw = String(text || "")
      .toLowerCase()
      .replace(/,/g, "")
      .replace(/×|x\b/g, " times ")
      .replace(/÷/g, " divided by ")
      .replace(/percent of/g, " percentof ")
      .replace(/%/g, " percent ")
      .replace(/to the power of/g, " ^ ")
      .replace(/squared/g, " ^ 2 ")
      .replace(/cubed/g, " ^ 3 ")
      .replace(/square root of/g, " sqrt ")
      .replace(/cubed root of|cube root of/g, " cbrt ")
      .replace(/multiplied by/g, " * ")
      .replace(/divided by/g, " / ")
      .replace(/plus/g, " + ")
      .replace(/minus/g, " - ")
      .replace(/times/g, " * ");
    var parts = raw.split(/(\s+)/);
    var out = [];
    var buf = [];
    function flush() {
      if (!buf.length) return;
      var n = wordsToNumber(buf);
      out.push(n == null ? buf.join(" ") : String(n));
      buf = [];
    }
    parts.forEach(function (p) {
      var t = p.trim();
      if (!t) return;
      if (WORDS[t] != null || t === "and") buf.push(t);
      else {
        flush();
        out.push(t);
      }
    });
    flush();
    return out.join(" ");
  }

  function tokenize(expr) {
    var s = expr.replace(/\s+/g, "");
    var tokens = [];
    var i = 0;
    while (i < s.length) {
      var ch = s[i];
      if (/[0-9.]/.test(ch)) {
        var num = "";
        while (i < s.length && /[0-9.]/.test(s[i])) num += s[i++];
        tokens.push({ t: "num", v: parseFloat(num) });
        continue;
      }
      if ("+-*/^%(),".indexOf(ch) !== -1) {
        tokens.push({ t: ch });
        i += 1;
        continue;
      }
      if (/[a-z]/i.test(ch)) {
        var id = "";
        while (i < s.length && /[a-z]/i.test(s[i])) id += s[i++];
        tokens.push({ t: "id", v: id.toLowerCase() });
        continue;
      }
      return null;
    }
    return tokens;
  }

  function parseExpr(tokens) {
    if (!tokens) return null;
    var i = 0;
    function peek() { return tokens[i] || { t: "end" }; }
    function eat(kind) {
      if (peek().t === kind) { i += 1; return true; }
      return false;
    }
    function parsePrimary() {
      if (peek().t === "num") { var n = peek().v; i += 1; return n; }
      if (peek().t === "id") {
        var name = peek().v;
        i += 1;
        if (name === "pi") return Math.PI;
        if (name === "e") return Math.E;
        if (eat("(")) {
          var arg = parseAdd();
          eat(")");
          if (name === "sqrt") return Math.sqrt(arg);
          if (name === "cbrt") return Math.cbrt(arg);
          if (name === "sin") return Math.sin(arg);
          if (name === "cos") return Math.cos(arg);
          if (name === "tan") return Math.tan(arg);
          if (name === "abs") return Math.abs(arg);
          if (name === "log") return Math.log10(arg);
          if (name === "ln") return Math.log(arg);
          if (name === "round") return Math.round(arg);
          if (name === "floor") return Math.floor(arg);
          if (name === "ceil") return Math.ceil(arg);
          return null;
        }
        return null;
      }
      if (eat("(")) {
        var inner = parseAdd();
        eat(")");
        return inner;
      }
      if (eat("-")) return -parsePrimary();
      if (eat("+")) return parsePrimary();
      return null;
    }
    function parsePow() {
      var left = parsePrimary();
      if (left == null) return null;
      if (eat("^")) {
        var right = parsePow();
        if (right == null) return null;
        return Math.pow(left, right);
      }
      return left;
    }
    function parseMul() {
      var left = parsePow();
      if (left == null) return null;
      while (peek().t === "*" || peek().t === "/" || peek().t === "%") {
        var op = peek().t;
        i += 1;
        var right = parsePow();
        if (right == null) return null;
        if (op === "*") left *= right;
        else if (op === "/") left /= right;
        else left %= right;
      }
      return left;
    }
    function parseAdd() {
      var left = parseMul();
      if (left == null) return null;
      while (peek().t === "+" || peek().t === "-") {
        var op = peek().t;
        i += 1;
        var right = parseMul();
        if (right == null) return null;
        left = op === "+" ? left + right : left - right;
      }
      return left;
    }
    var val = parseAdd();
    if (val == null || i < tokens.length) return null;
    return val;
  }

  function evalText(text) {
    var rewritten = rewriteWords(text);
    var pct = rewritten.match(/(-?\d+(?:\.\d+)?)\s+percentof\s+(-?\d+(?:\.\d+)?)/);
    if (pct) return (parseFloat(pct[1]) / 100) * parseFloat(pct[2]);
    var cleaned = rewritten
      .replace(/what(?:'s| is)|calculate|compute|equals?|cuánto es|combien|was ist/gi, " ")
      .replace(/[^0-9a-z+\-*/^%().,\s]/gi, " ");
    var expr = cleaned.replace(/percent/g, " * 0.01 ").replace(/\s+/g, "");
    if (!expr || expr.length > 120) return null;
    return parseExpr(tokenize(expr));
  }

  function convertTemp(n, from, to) {
    var c = n;
    if (from === "f") c = (n - 32) * (5 / 9);
    if (from === "k") c = n - 273.15;
    if (to === "c") return c;
    if (to === "f") return c * (9 / 5) + 32;
    if (to === "k") return c + 273.15;
    return null;
  }

  function convert(text) {
    var n = String(text || "").toLowerCase();
    var m = n.match(/(-?\d+(?:\.\d+)?)\s*([a-z]+)\s+(?:to|in|into)\s+([a-z]+)/);
    if (!m) return null;
    var from = UNITS[m[2]];
    var to = UNITS[m[3]];
    if (!from || !to || from.dim !== to.dim) return null;
    var val = parseFloat(m[1]);
    var out;
    if (from.dim === "temp") out = convertTemp(val, from.to, to.to);
    else out = (val * from.to) / to.to;
    if (out == null || !isFinite(out)) return null;
    return { value: out, from: m[2], to: m[3], input: val };
  }

  function looksMath(text) {
    var n = String(text || "").toLowerCase();
    if (convert(n)) return true;
    if (/\b(percent of|square root|times|divided by|plus|minus|to the power)\b/.test(n) && /\d|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty/.test(n)) return true;
    if (/[\d)]\s*[+\-*/^x×÷]\s*[\d(]/.test(n)) return true;
    if (/\b(calculate|compute|what(?:'s| is))\b/.test(n) && /\d/.test(n)) return true;
    return false;
  }

  function answer(text) {
    var conv = convert(text);
    if (conv) {
      return {
        kind: "convert",
        value: conv.value,
        reply: nice(conv.input) + " " + conv.from + " is about " + nice(conv.value) + " " + conv.to + ". =)",
      };
    }
    var v = evalText(text);
    if (v == null || !isFinite(v)) return null;
    return { kind: "calc", value: v, reply: "That’s " + nice(v) + ". =)" };
  }

  var api = {
    evalText: evalText,
    convert: convert,
    looksMath: looksMath,
    answer: answer,
    nice: nice,
    rewriteWords: rewriteWords,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantMath = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
