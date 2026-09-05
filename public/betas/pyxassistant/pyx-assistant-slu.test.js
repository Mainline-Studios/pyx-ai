/**
 * Pyx Assistant SLU tests — run with:
 *   node public/betas/pyxassistant/pyx-assistant-slu.test.js
 */
"use strict";

var slu = require("./pyx-assistant-slu.js");
var i18n = require("./pyx-assistant-i18n.js");

var failed = 0;

function assert(cond, msg) {
  if (!cond) {
    failed += 1;
    console.error("FAIL  " + msg);
  } else {
    console.log("ok    " + msg);
  }
}

var golden = slu.evaluateGolden();
assert(golden.accuracy === 1, "golden set accuracy is 100% (" + golden.hit + "/" + golden.total + ")");
if (golden.misses.length) {
  golden.misses.forEach(function (m) {
    console.error("      ", m.text, "expected", m.expected, "got", m.got);
  });
}

assert(slu.classify("hello").intent === "greet", "hello → greet");
assert(slu.classify("what's 9 * 8").intent === "calculator", "math → calculator");
assert(slu.safeEvalMath("9*8") === 72, "9*8 = 72");
assert(slu.extractTheme("switch to mint") === "mint", "theme slot mint");
assert(slu.extractLang("speak Spanish") === "es", "language slot es");

var resolved = slu.resolve(slu.classify("what time is it"), {
  lang: "en",
  t: i18n.t,
});
assert(typeof resolved.reply === "string" && resolved.reply.indexOf("It’s") === 0, "time reply is local");
assert(resolved.useLlm === false, "time does not call the LLM");

var chat = slu.resolve(slu.classify("explain gravity"), { lang: "en", t: i18n.t });
assert(chat.useLlm === true, "open questions use the LLM");

var weather = slu.resolve(slu.classify("weather in paris"), { lang: "en", t: i18n.t });
assert(weather.useWeb === true, "weather requests web");

assert(i18n.t("en", "name") === "Pyx Assistant", "product name is Pyx Assistant");
assert(i18n.t("es", "name") === "Pyx Assistant", "name stays Pyx Assistant in ES");

if (failed) {
  console.error("\n" + failed + " failed");
  process.exit(1);
}
console.log("\nAll SLU tests passed.");
