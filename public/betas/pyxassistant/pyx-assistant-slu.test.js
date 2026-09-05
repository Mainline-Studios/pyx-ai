/**
 * Pyx Assistant tests — run with:
 *   node public/betas/pyxassistant/pyx-assistant-slu.test.js
 */
"use strict";

var slu = require("./pyx-assistant-slu.js");
var i18n = require("./pyx-assistant-i18n.js");
var math = require("./pyx-assistant-math.js");
var kb = require("./pyx-assistant-kb.js");
var data = require("./kb/pyx-assistant-kb.json");

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
assert(slu.classify("tell me a joke").intent === "joke", "joke intent");
assert(slu.classify("fun fact").intent === "fact", "fact intent");
assert(math.evalText("9*8") === 72, "9*8 = 72");
assert(math.evalText("fifteen percent of 80") === 12, "15% of 80 = 12");
assert(math.evalText("square root of 144") === 12, "sqrt 144");
assert(math.evalText("twenty plus twenty two") === 42, "word numbers");
assert(math.answer("32 f to c").value === 0, "32 F → 0 C");
assert(math.answer("5 km to miles").value > 3 && math.answer("5 km to miles").value < 3.2, "5 km → miles");
assert(math.looksMath("what's 12 times 4") === true, "looksMath times");
assert(slu.extractTheme("switch to mint") === "mint", "theme slot mint");
assert(slu.extractLang("speak Spanish") === "es", "language slot es");

var resolved = slu.resolve(slu.classify("what time is it"), { lang: "en", t: i18n.t });
assert(typeof resolved.reply === "string" && resolved.reply.indexOf("It’s") === 0, "time reply is local");
assert(resolved.useLlm === false, "time does not call Talk");

var joke = slu.resolve(slu.classify("tell me a joke"), { lang: "en", t: i18n.t });
assert(joke.special === "__JOKE__", "jokes are local specials");
assert(joke.useLlm === false, "jokes do not call Talk");

var chat = slu.resolve(slu.classify("explain gravity"), { lang: "en", t: i18n.t });
assert(chat.useLlm === false, "open questions do not call Talk");

var weather = slu.resolve(slu.classify("weather in paris"), { lang: "en", t: i18n.t });
assert(weather.useWeb === false, "weather stays local");
assert(/weather app|live weather/i.test(weather.reply), "weather is honest about no API");

var n = kb.load(data);
assert(n >= 1000, "knowledge base has 1000+ records (" + n + ")");
var grav = kb.retrieve("what is gravity", 0.4);
assert(grav && /gravity/.test(grav.rec.q) && grav.reply.length > 20, "retrieve gravity");
var photo = kb.retrieve("explain photosynthesis in simple terms", 0.35);
assert(photo && /photosynthesis/.test(photo.rec.q) && /light|plant/i.test(photo.reply), "retrieve photosynthesis");
assert(/honey|bees/i.test(kb.retrieve("fact about honey", 0.4).reply), "retrieve honey fact");
var j1 = kb.expandSpecial("__JOKE__");
var j2 = kb.expandSpecial("__JOKE__");
assert(j1 && j2 && j1.length > 10, "jokes resolve from the pack");
var haiku = kb.retrieve("write a haiku about rain", 0.62);
assert(!haiku || /rain|orb|pastel/i.test(haiku.reply), "haiku does not steal the email how-to");
assert(kb.warmFallback("write a haiku about rain").length > 20, "unmatched still gets a warm reply");
assert(i18n.t("en", "name") === "Pyx Assistant", "product name is Pyx Assistant");
assert(i18n.t("es", "name") === "Pyx Assistant", "name stays Pyx Assistant in ES");
assert(/on-device|local notebook|no cloud/i.test(i18n.t("en", "identity")), "identity says on-device");

if (failed) {
  console.error("\n" + failed + " failed");
  process.exit(1);
}
console.log("\nAll assistant tests passed.");
